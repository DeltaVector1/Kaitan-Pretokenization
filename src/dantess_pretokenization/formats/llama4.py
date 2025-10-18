"""Llama4 format definition."""

from .base import BaseChatFormat


class Llama4Format(BaseChatFormat):
    """Llama4 chat format."""

    @property
    def name(self) -> str:
        return "Llama4"

    @property
    def format_tokens(self) -> dict:
        return {
            "starting_sequence": {"text": "<|begin_of_text|>", "tokens": None},
            "system_prefix": {"text": "<|header_start|>system<|header_end|>\n\n", "tokens": None},
            "user_prefix": {"text": "<|header_start|>user<|header_end|>\n\n", "tokens": None},
            "assistant_prefix": {"text": "<|header_start|>assistant<|header_end|>\n\n", "tokens": None},
            "tool_prefix": {"text": "<|header_start|>system<|header_end|>\n\n", "tokens": None},
            "turn_seperator": {"text": "", "tokens": None},
            "turn_suffix": {"text": "<|eot|>", "tokens": None},
        }
