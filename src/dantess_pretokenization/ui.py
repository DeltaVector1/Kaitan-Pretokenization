"""TUI components for pretokenization."""

import logging

import questionary
from colorama import Fore, Style, init

logger = logging.getLogger(__name__)


def select_chat_format(available_formats: list[str]) -> str:
    """Show TUI to select a chat format.

    Args:
        available_formats: List of available format names

    Returns:
        Selected format name
    """
    init()  # Initialize colorama

    answer = questionary.select(
        "What chat format do you want?",
        choices=available_formats,
        style=questionary.Style(
            [
                ("qmark", "fg:cyan bold"),
                ("question", "bold"),
                ("answer", "fg:green bold"),
                ("pointer", "fg:cyan bold"),
                ("highlighted", "fg:cyan bold"),
                ("selected", "fg:green"),
            ]
        ),
    ).ask()

    return answer


def show_colored_preview(input_ids: list[int], labels: list[int], tokenizer, mask_token_id: int = -100) -> None:
    """Show a colored preview of tokenization.

    Args:
        input_ids: List of token IDs
        labels: List of label IDs (masked or unmasked)
        tokenizer: HuggingFace tokenizer
        mask_token_id: ID used for masked tokens (default: -100)
    """
    init()  # Initialize colorama

    decoded_tokens = []
    for i, token_id in enumerate(input_ids):
        token = tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        color = Fore.GREEN if labels[i] != mask_token_id else Fore.RED
        decoded_tokens.append(f"{color}{token}{Style.RESET_ALL}")

    print("\n" + "=" * 80)
    print("TOKENIZATION PREVIEW")
    print("=" * 80)
    print(f"{Fore.GREEN}Green = Unmasked (trained){Style.RESET_ALL}")
    print(f"{Fore.RED}Red = Masked (not trained){Style.RESET_ALL}")
    print("=" * 80)
    print("".join(decoded_tokens))
    print("=" * 80)
    print(f"\nTotal tokens: {len(input_ids)}")
    print(f"Unmasked tokens: {sum(1 for label in labels if label != mask_token_id)}")
    print(f"Masked tokens: {sum(1 for label in labels if label == mask_token_id)}")
    print("=" * 80 + "\n")


def confirm_tokenization() -> bool:
    """Ask user to confirm before starting tokenization.

    Returns:
        True if user confirms, False otherwise
    """
    answer = questionary.confirm(
        "Do you want to proceed with tokenization?",
        default=True,
        style=questionary.Style(
            [
                ("qmark", "fg:yellow bold"),
                ("question", "bold"),
                ("answer", "fg:green bold"),
            ]
        ),
    ).ask()

    return answer if answer is not None else False
