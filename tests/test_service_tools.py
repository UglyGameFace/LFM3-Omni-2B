import sys
from pathlib import Path
from unittest import TestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from service.tools import (ToolValidationError, execute_tool, tool_definitions,
                           validate_args, validate_tool_definition)


class ToolTests(TestCase):
    def test_add_executes_deterministically(self):
        self.assertEqual(execute_tool("add", {"a": 2, "b": 3}), {"result": 5})

    def test_echo_and_clock(self):
        self.assertEqual(execute_tool("echo", {"text": "hi"}), {"echo": "hi"})
        self.assertIn("utc_now", execute_tool("clock", {}))

    def test_unregistered_tool_never_executes(self):
        with self.assertRaises(ToolValidationError):
            execute_tool("delete_everything", {})

    def test_wrong_types_and_missing_fields_rejected(self):
        with self.assertRaises(ToolValidationError):
            execute_tool("add", {"a": "two", "b": 3})
        with self.assertRaises(ToolValidationError):
            execute_tool("add", {"a": 1})
        with self.assertRaises(ToolValidationError):
            execute_tool("add", {"a": 1, "b": 2, "c": 3})

    def test_schema_validator_covers_types_enums_and_ranges(self):
        schema = {"type": "object",
                  "properties": {"n": {"type": "integer", "minimum": 0, "maximum": 10},
                                 "s": {"type": "string", "enum": ["a", "b"]}},
                  "required": ["n"]}
        validate_args({"n": 5, "s": "a"}, schema)
        for bad in ({"n": 11}, {"n": 5, "s": "c"}, {"s": "a"}, {"n": 1.5}):
            with self.assertRaises(ToolValidationError, msg=str(bad)):
                validate_args(bad, schema)

    def test_tool_definition_validation(self):
        schema = validate_tool_definition(tool_definitions()[0])
        self.assertEqual(schema["type"], "object")
        with self.assertRaises(ToolValidationError):
            validate_tool_definition({"type": "chat", "function": {"name": "x"}})
        with self.assertRaises(ToolValidationError):
            validate_tool_definition({"type": "function",
                                      "function": {"name": "", "parameters": {}}})
        with self.assertRaises(ToolValidationError):
            validate_tool_definition({"type": "function",
                                      "function": {"name": "x",
                                                   "parameters": {"type": "array"}}})

    def test_registry_is_small_and_deterministic(self):
        names = {d["function"]["name"] for d in tool_definitions()}
        self.assertEqual(names, {"echo", "add", "clock"})
