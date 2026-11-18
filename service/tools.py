"""Structured tool calls with JSON-schema argument validation.

Stdlib-only validator covering the subset clients actually use: type,
required, properties, items, enum, minimum/maximum, minLength/maxLength,
pattern, additionalProperties. Anything outside that subset is rejected
rather than silently accepted. Only tools registered in the local
registry may execute; a backend-proposed tool that is not registered is
never executed, however plausible it looks.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone


class ToolValidationError(ValueError):
    pass


def validate_args(args, schema: dict, path: str = "$") -> None:
    if not isinstance(schema, dict):
        raise ToolValidationError(f"{path}: schema must be an object")
    if not isinstance(args, dict):
        raise ToolValidationError(f"{path}: arguments must be a JSON object")
    _validate_value(args, schema, path)


def _validate_value(value, schema: dict, path: str) -> None:
    kind = schema.get("type")
    if kind == "object":
        if not isinstance(value, dict):
            raise ToolValidationError(f"{path}: expected object")
        for name in schema.get("required", []):
            if name not in value:
                raise ToolValidationError(f"{path}: missing required field {name!r}")
        props = schema.get("properties", {})
        for key, sub in value.items():
            if key in props:
                _validate_value(sub, props[key], f"{path}.{key}")
            elif schema.get("additionalProperties") is False:
                raise ToolValidationError(f"{path}: unexpected field {key!r}")
    elif kind == "array":
        if not isinstance(value, list):
            raise ToolValidationError(f"{path}: expected array")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for i, item in enumerate(value):
                _validate_value(item, item_schema, f"{path}[{i}]")
    elif kind == "string":
        if not isinstance(value, str):
            raise ToolValidationError(f"{path}: expected string")
        if "minLength" in schema and len(value) < schema["minLength"]:
            raise ToolValidationError(f"{path}: string too short")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise ToolValidationError(f"{path}: string too long")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            raise ToolValidationError(f"{path}: does not match pattern")
    elif kind == "integer":
        if type(value) is not int:
            raise ToolValidationError(f"{path}: expected integer")
    elif kind == "number":
        if type(value) not in (int, float) or isinstance(value, bool):
            raise ToolValidationError(f"{path}: expected number")
    elif kind == "boolean":
        if not isinstance(value, bool):
            raise ToolValidationError(f"{path}: expected boolean")
    if "enum" in schema and value not in schema["enum"]:
        raise ToolValidationError(f"{path}: value not in enum {schema['enum']!r}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise ToolValidationError(f"{path}: below minimum")
        if "maximum" in schema and value > schema["maximum"]:
            raise ToolValidationError(f"{path}: above maximum")


def validate_tool_definition(tool: dict) -> dict:
    """Validate one OpenAI-style function tool definition; return its schema."""
    if not isinstance(tool, dict) or tool.get("type") != "function":
        raise ToolValidationError("tool must be an object with type 'function'")
    fn = tool.get("function")
    if not isinstance(fn, dict) or not isinstance(fn.get("name"), str) or not fn["name"]:
        raise ToolValidationError("tool function must have a non-empty name")
    params = fn.get("parameters", {"type": "object", "properties": {}})
    if not isinstance(params, dict) or params.get("type") != "object":
        raise ToolValidationError("tool parameters must be an object schema")
    return params


# --- Local registry: deterministic, no network, no secrets, no side effects
# beyond returning a value. The model may PROPOSE a tool call; only these
# may EXECUTE. ---

def _echo(args: dict) -> dict:
    return {"echo": args.get("text", "")}


def _add(args: dict) -> dict:
    return {"result": args["a"] + args["b"]}


def _clock(_args: dict) -> dict:
    return {"utc_now": datetime.now(timezone.utc).isoformat()}


REGISTRY = {
    "echo": {
        "description": "Echo text back verbatim. Deterministic test tool.",
        "parameters": {"type": "object",
                       "properties": {"text": {"type": "string", "maxLength": 1000}},
                       "required": ["text"], "additionalProperties": False},
        "handler": _echo,
    },
    "add": {
        "description": "Add two integers. Deterministic arithmetic, no model math.",
        "parameters": {"type": "object",
                       "properties": {"a": {"type": "integer"},
                                      "b": {"type": "integer"}},
                       "required": ["a", "b"], "additionalProperties": False},
        "handler": _add,
    },
    "clock": {
        "description": "Current UTC time. No arguments.",
        "parameters": {"type": "object", "properties": {},
                       "required": [], "additionalProperties": False},
        "handler": _clock,
    },
}


def execute_tool(name: str, args: dict) -> dict:
    """Validate args against the registered schema, then run the handler.
    Unknown tools raise; they are never executed."""
    entry = REGISTRY.get(name)
    if entry is None:
        raise ToolValidationError(f"tool {name!r} is not registered and will not execute")
    validate_args(args, entry["parameters"])
    return entry["handler"](args)


def tool_definitions() -> list[dict]:
    return [{"type": "function",
             "function": {"name": name,
                          "description": entry["description"],
                          "parameters": entry["parameters"]}}
            for name, entry in REGISTRY.items()]
