"""ChatML format definition."""

from .base import BaseChatFormat


class ChatMLFormat(BaseChatFormat):
    """ChatML format (standard implementation without BOS fix)."""

    @property
    def name(self) -> str:
        return "ChatML"

    @property
    def format_tokens(self) -> dict:
        return {
            "starting_sequence": {"text": "", "tokens": None},
            "system_prefix": {"text": "<|im_start|>system\n", "tokens": None},
            "user_prefix": {"text": "<|im_start|>user\n", "tokens": None},
            "assistant_prefix": {"text": "<|im_start|>assistant\n", "tokens": None},
            "tool_prefix": {"text": "<|im_start|>tool\n", "tokens": None},
            "turn_seperator": {"text": "\n", "tokens": None},
            "turn_suffix": {"text": "<|im_end|>", "tokens": None},
        }
