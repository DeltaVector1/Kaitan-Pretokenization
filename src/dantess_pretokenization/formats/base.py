"""Base class for chat format definitions."""

from abc import ABC, abstractmethod
from typing import Any


class BaseChatFormat(ABC):
    """Base class for chat format definitions."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the format."""
        pass

    @property
    @abstractmethod
    def format_tokens(self) -> dict[str, dict[str, Any]]:
        """Return the format token configuration."""
        pass

    @property
    def system_from_values(self) -> list[str]:
        """Return the list of valid system role identifiers."""
        return ["system", "sys"]

    @property
    def user_from_values(self) -> list[str]:
        """Return the list of valid user role identifiers."""
        return ["user", "human"]

    @property
    def assistant_from_values(self) -> list[str]:
        """Return the list of valid assistant role identifiers."""
        return ["assistant", "model", "gpt"]

    @property
    def tool_from_values(self) -> list[str]:
        """Return the list of valid tool role identifiers."""
        return ["tool", "function", "environment"]

    @property
    def mask_token_id(self) -> int:
        """Return the mask token ID."""
        return -100
