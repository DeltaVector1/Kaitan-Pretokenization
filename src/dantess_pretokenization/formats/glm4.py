"""GLM4 format definition."""

from .base import BaseChatFormat


class GLM4Format(BaseChatFormat):
    """GLM4 chat format."""

    @property
    def name(self) -> str:
        return "GLM4"

    @property
    def format_tokens(self) -> dict:
        return {
            "starting_sequence": {"text": "[gMASK]<sop>", "tokens": None},
            "system_prefix": {"text": "<|system|>\n", "tokens": None},
            "user_prefix": {"text": "<|user|>\n", "tokens": None},
            "user_prefix_bare": {"text": "\n", "tokens": None},
            "assistant_prefix": {"text": "<|assistant|>\n", "tokens": None},
            "tool_prefix": {"text": "<|system|>\n", "tokens": None},
            "turn_seperator": {"text": "", "tokens": None},
            "turn_suffix": {"text": "", "tokens": None},
            "turn_suffix_assistant": {"text": "<|user|>", "tokens": None},
        }
