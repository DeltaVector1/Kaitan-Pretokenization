"""Llama3 format definition."""

from .base import BaseChatFormat


class Llama3Format(BaseChatFormat):
    """Llama3 chat format."""

    @property
    def name(self) -> str:
        return "Llama3"

    @property
    def format_tokens(self) -> dict:
        return {
            "starting_sequence": {"text": "<|begin_of_text|>", "tokens": None},
            "system_prefix": {"text": "<|start_header_id|>system<|end_header_id|>\n\n", "tokens": None},
            "user_prefix": {"text": "<|start_header_id|>user<|end_header_id|>\n\n", "tokens": None},
            "assistant_prefix": {"text": "<|start_header_id|>assistant<|end_header_id|>\n\n", "tokens": None},
            "tool_prefix": {"text": "<|start_header_id|>system<|end_header_id|>\n\n", "tokens": None},
            "turn_seperator": {"text": "", "tokens": None},
            "turn_suffix": {"text": "<|eot_id|>", "tokens": None},
        }
