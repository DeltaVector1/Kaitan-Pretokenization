"""Mistral V7 Tekken format definition."""

from .base import BaseChatFormat


class MistralV7TekkenFormat(BaseChatFormat):
    """Mistral V7 Tekken chat format."""

    @property
    def name(self) -> str:
        return "Mistral V7 Tekken"

    @property
    def format_tokens(self) -> dict:
        return {
            "starting_sequence": {"text": "<s>", "tokens": None},
            "system_prefix": {"text": "[SYSTEM_PROMPT]", "tokens": None},
            "user_prefix": {"text": "[INST]", "tokens": None},
            "assistant_prefix": {"text": "", "tokens": None},
            "tool_prefix": {"text": "[TOOL_RESULTS]", "tokens": None},
            "turn_seperator": {"text": "", "tokens": None},
            "turn_suffix": {"text": "", "tokens": None},
            "turn_suffix_system": {"text": "[/SYSTEM_PROMPT]", "tokens": None},
            "turn_suffix_user": {"text": "[/INST]", "tokens": None},
            "turn_suffix_assistant": {"text": "</s>", "tokens": None},
            "turn_suffix_tool": {"text": "[/TOOL_RESULTS]", "tokens": None},
        }
