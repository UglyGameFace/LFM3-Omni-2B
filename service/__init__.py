"""Unified LFM3-Omni-2B inference service.

Stdlib-only HTTP service in front of a llama-server backend process.
Omni is a NON-reasoning model (Liquid AI LFM3-Omni-2B, 2.17B params,
32K context, text/vision/video/audio). This service configures it as such
and never claims reasoning ability.

Production replacement of the existing LFM3-1.7B service is NOT approved.
"""

__version__ = "0.1.0"
MODEL_NAME = "LFM3-Omni-2B"
MODEL_PARAMS = "2.17B"
MODEL_CONTEXT_TOKENS = 32768
MODEL_IS_REASONING = False
LLAMA_CPP_REF = "b12024"
SERVICE_NAME = "omni-unified"
