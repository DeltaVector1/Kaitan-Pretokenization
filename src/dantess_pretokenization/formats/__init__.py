"""Chat format definitions for pretokenization."""

from typing import Any

from .chatml import ChatMLFormat
from .chatml_bos import ChatMLBOSFormat
from .glm4 import GLM4Format
from .llama3 import Llama3Format
from .llama4 import Llama4Format
from .mistral_v7_tekken import MistralV7TekkenFormat

# Registry of all available formats
FORMATS: dict[str, Any] = {
    "ChatML": ChatMLFormat,
    "ChatML with BOS Fix": ChatMLBOSFormat,
    "GLM4": GLM4Format,
    "Llama3": Llama3Format,
    "Llama4": Llama4Format,
    "Mistral V7 Tekken": MistralV7TekkenFormat,
}

__all__ = [
    "FORMATS",
    "ChatMLFormat",
    "ChatMLBOSFormat",
    "GLM4Format",
    "Llama3Format",
    "Llama4Format",
    "MistralV7TekkenFormat",
]
