"""ChatML format with BOS token fix."""

from .base import BaseChatFormat


class ChatMLBOSFormat(BaseChatFormat):
    """ChatML format with proper BOS token handling."""

    @property
    def name(self) -> str:
        return "ChatML with BOS Fix"

    @property
    def format_tokens(self) -> dict:
        return {
            "starting_sequence": {
                "text": "",
                "tokens": None,  # Will be filled with BOS token by tokenizer
            },
            "system_prefix": {"text": "<|im_start|>system\n", "tokens": None},
            "user_prefix": {"text": "<|im_start|>user\n", "tokens": None},
            "assistant_prefix": {"text": "<|im_start|>assistant\n", "tokens": None},
            "tool_prefix": {"text": "<|im_start|>tool\n", "tokens": None},
            "turn_seperator": {"text": "\n", "tokens": None},
            "turn_suffix": {"text": "<|im_end|>", "tokens": None},
        }
