"""Tokenization logic for pretokenization."""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from datasets import load_dataset

logger = logging.getLogger(__name__)


def fill_format_tokens(tokenizer: Any, format_tokens: dict[str, dict[str, Any]], is_bos_fix: bool = False) -> None:
    """Fill the format tokens with their tokenized values.

    Args:
        tokenizer: The HuggingFace tokenizer
        format_tokens: Dictionary of format tokens to fill
        is_bos_fix: Whether to use BOS token for starting sequence
    """
    # First, handle the BOS token for starting_sequence if BOS fix is enabled
    if is_bos_fix and format_tokens["starting_sequence"]["tokens"] is None:
        if tokenizer.bos_token_id is not None:
            format_tokens["starting_sequence"]["tokens"] = [tokenizer.bos_token_id]
            logger.debug(f"Set starting_sequence to BOS token: {tokenizer.bos_token_id}")
        else:
            format_tokens["starting_sequence"]["tokens"] = []
            logger.warning("No BOS token found in tokenizer, starting_sequence will be empty")

    # Tokenize all non-separator tokens
    for key, value in format_tokens.items():
        if key not in ["turn_seperator", "starting_sequence"] and value["tokens"] is None:
            tokens = tokenizer.encode(value["text"], add_special_tokens=False)
            format_tokens[key]["tokens"] = tokens
            logger.debug(f"Tokenized {key}: {value['text']} -> {tokens}")

    # Handle starting_sequence if not BOS fix
    if not is_bos_fix and format_tokens["starting_sequence"]["tokens"] is None:
        tokens = tokenizer.encode(format_tokens["starting_sequence"]["text"], add_special_tokens=False)
        format_tokens["starting_sequence"]["tokens"] = tokens
        logger.debug(f"Tokenized starting_sequence: {format_tokens['starting_sequence']['text']} -> {tokens}")

    # Special handling for the turn separator
    if format_tokens["turn_seperator"]["tokens"] is None:
        # Get suffix and prefix for assistant (most common case)
        suffix_key = "turn_suffix"
        prefix_key = "assistant_prefix"

        combined_text = (
            format_tokens[suffix_key]["text"]
            + format_tokens["turn_seperator"]["text"]
            + format_tokens[prefix_key]["text"]
        )

        combined_tokens = tokenizer.encode(combined_text, add_special_tokens=False)
        suffix_tokens = format_tokens[suffix_key]["tokens"]
        prefix_tokens = format_tokens[prefix_key]["tokens"]

        # Calculate separator tokens
        if suffix_tokens and prefix_tokens:
            if (
                combined_tokens[: len(suffix_tokens)] == suffix_tokens
                and combined_tokens[-len(prefix_tokens) :] == prefix_tokens
            ):
                separator_tokens = combined_tokens[len(suffix_tokens) : -len(prefix_tokens)]
            else:
                separator_tokens = []
        else:
            separator_tokens = combined_tokens

        format_tokens["turn_seperator"]["tokens"] = separator_tokens
        logger.debug(f"Tokenized turn_seperator: {format_tokens['turn_seperator']['text']} -> {separator_tokens}")


def normalize_conversation(item: list[dict], format_obj: Any) -> list[dict]:
    """Normalize the conversation format.

    Args:
        item: List of conversation turns
        format_obj: Format object with role value lists

    Returns:
        Normalized conversation turns
    """
    for turn in item:
        from_value = turn["from"].lower()

        if from_value in format_obj.system_from_values:
            turn["from"] = "system"
        elif from_value in format_obj.user_from_values:
            turn["from"] = "human"
        elif from_value in format_obj.assistant_from_values:
            turn["from"] = "gpt"
        elif from_value in format_obj.tool_from_values:
            turn["from"] = "tool"

        if "loss" not in turn or turn["loss"] is None:
            turn["loss"] = turn["from"] == "gpt"

    return item


def tokenize_item(
    item: dict, tokenizer: Any, format_tokens: dict, format_obj: Any, max_length: int
) -> dict[str, list] | None:
    """Tokenize a single conversation item.

    Args:
        item: Conversation item with 'conversations' field
        tokenizer: HuggingFace tokenizer
        format_tokens: Dictionary of format tokens
        format_obj: Format object with role definitions
        max_length: Maximum sequence length

    Returns:
        Dictionary with 'input_ids', 'attention_mask', and 'labels', or None if invalid
    """
    mask_token_id = format_obj.mask_token_id
    turns = normalize_conversation(item["conversations"], format_obj)

    starting_sequence_tokens = format_tokens["starting_sequence"]["tokens"]
    input_ids = list(starting_sequence_tokens)
    labels = [mask_token_id] * len(starting_sequence_tokens)

    turns_data = []
    preceding_turn = "none"

    for turn_idx, turn in enumerate(turns):
        turn_tokenized = {"loss": turn.get("loss", False), "input_ids": [], "labels": []}

        from_value = turn["from"].lower()
        should_contribute_to_loss = turn.get("loss", False)

        # Select role tokens
        role_tokens = None
        if from_value in format_obj.system_from_values:
            role_tokens = format_tokens["system_prefix"]["tokens"]
        elif from_value in format_obj.user_from_values:
            # GLM4 has special handling for user prefix
            if "user_prefix_bare" in format_tokens and preceding_turn == "assistant":
                role_tokens = format_tokens["user_prefix_bare"]["tokens"]
            else:
                role_tokens = format_tokens["user_prefix"]["tokens"]
        elif from_value in format_obj.assistant_from_values:
            role_tokens = format_tokens["assistant_prefix"]["tokens"]
        elif from_value in format_obj.tool_from_values:
            role_tokens = format_tokens["tool_prefix"]["tokens"]
        else:
            raise ValueError(f"Unknown role '{from_value}' in turn {turn_idx}.")

        turn_tokenized["input_ids"].extend(role_tokens)
        turn_tokenized["labels"].extend([mask_token_id] * len(role_tokens))

        # Get turn value
        value = turn["value"]
        if (not isinstance(value, str) or len(value) == 0) and from_value in format_obj.assistant_from_values:
            logger.debug(f"Skipping turn {turn_idx} with invalid value: {value}")
            return None

        # Handle prefix
        turn_prefix = turn.get("prefix", "")
        if turn_prefix:
            prefix_tokens = tokenizer.encode(turn_prefix, add_special_tokens=False)
            turn_tokenized["input_ids"].extend(prefix_tokens)
            turn_tokenized["labels"].extend([mask_token_id] * len(prefix_tokens))

        # Tokenize value
        value_tokens = tokenizer.encode(value, add_special_tokens=False)
        turn_tokenized["input_ids"].extend(value_tokens)

        if should_contribute_to_loss:
            turn_tokenized["labels"].extend(value_tokens)
        else:
            turn_tokenized["labels"].extend([mask_token_id] * len(value_tokens))

        # Add turn suffix (format-specific)
        suffix_key = "turn_suffix"
        if from_value in format_obj.assistant_from_values and "turn_suffix_assistant" in format_tokens:
            suffix_key = "turn_suffix_assistant"
        elif from_value in format_obj.system_from_values and "turn_suffix_system" in format_tokens:
            suffix_key = "turn_suffix_system"
        elif from_value in format_obj.user_from_values and "turn_suffix_user" in format_tokens:
            suffix_key = "turn_suffix_user"
        elif from_value in format_obj.tool_from_values and "turn_suffix_tool" in format_tokens:
            suffix_key = "turn_suffix_tool"

        suffix_tokens = format_tokens[suffix_key]["tokens"]
        turn_tokenized["input_ids"].extend(suffix_tokens)

        if should_contribute_to_loss:
            turn_tokenized["labels"].extend(suffix_tokens)
        else:
            turn_tokenized["labels"].extend([mask_token_id] * len(suffix_tokens))

        # Check length
        separator_length = len(format_tokens["turn_seperator"]["tokens"]) if turns_data else 0
        current_length = len(input_ids)
        new_turn_length = len(turn_tokenized["input_ids"])

        if current_length + separator_length + new_turn_length > max_length:
            break

        turns_data.append(turn_tokenized)

        # Track preceding turn for GLM4
        if from_value in format_obj.system_from_values:
            preceding_turn = "system"
        elif from_value in format_obj.user_from_values:
            preceding_turn = "user"
        elif from_value in format_obj.assistant_from_values:
            preceding_turn = "assistant"
        elif from_value in format_obj.tool_from_values:
            preceding_turn = "tool"

    # Check if any turns contribute to loss
    if not any(turn["loss"] for turn in turns_data):
        return None

    # Remove turns after last loss turn
    last_loss_turn = max(i for i, turn in enumerate(turns_data) if turn["loss"])
    turns_data = turns_data[: last_loss_turn + 1]

    # Combine turns
    for turn_data in turns_data:
        input_ids.extend(turn_data["input_ids"])
        labels.extend(turn_data["labels"])

        if turn_data != turns_data[-1]:
            input_ids.extend(format_tokens["turn_seperator"]["tokens"])
            labels.extend([mask_token_id] * len(format_tokens["turn_seperator"]["tokens"]))

    attention_mask = [1] * len(input_ids)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def tokenize_dataset(
    dataset_path: str, tokenizer: Any, format_tokens: dict, format_obj: Any, max_length: int
) -> list[dict]:
    """Tokenize an entire dataset.

    Args:
        dataset_path: HuggingFace dataset path
        tokenizer: HuggingFace tokenizer
        format_tokens: Dictionary of format tokens
        format_obj: Format object
        max_length: Maximum sequence length

    Returns:
        List of tokenized items
    """
    logger.info(f"Loading dataset {dataset_path} in streaming mode...")
    try:
        loaded_data = load_dataset(dataset_path, split="train", streaming=True, trust_remote_code=True)
    except Exception as e:
        logger.warning(f"Failed loading {dataset_path} with trust_remote_code=True, trying without: {e}")
        loaded_data = load_dataset(dataset_path, split="train", streaming=True)

    logger.info(f"Successfully loaded {dataset_path} stream.")

    tokenized_items = []
    processed_count = 0
    start_time = time.time()
    log_interval = 10000

    with ThreadPoolExecutor() as executor:
        futures = []
        for item in loaded_data:
            future = executor.submit(tokenize_item, item, tokenizer, format_tokens, format_obj, max_length)
            futures.append(future)

            if len(futures) >= executor._max_workers * 2:
                while futures:
                    future = futures.pop(0)
                    try:
                        result = future.result()
                        if result is not None:
                            tokenized_items.append(result)
                        processed_count += 1
                        if processed_count % log_interval == 0:
                            elapsed = time.time() - start_time
                            logger.info(f"Processed {processed_count} items from {dataset_path}... ({elapsed:.2f}s)")
                    except Exception as exc:
                        logger.error(f"An item from {dataset_path} generated an exception during tokenization: {exc}")

        # Process remaining futures
        logger.info(f"Processing remaining {len(futures)} futures for {dataset_path}...")
        for future in futures:
            try:
                result = future.result()
                if result is not None:
                    tokenized_items.append(result)
                processed_count += 1
                if processed_count % log_interval == 0:
                    elapsed = time.time() - start_time
                    logger.info(f"Processed {processed_count} items from {dataset_path}... ({elapsed:.2f}s)")
            except Exception as exc:
                logger.error(f"An item from {dataset_path} generated an exception during final tokenization: {exc}")

    total_time = time.time() - start_time
    logger.info(f"Finished processing {processed_count} items from {dataset_path} in {total_time:.2f}s")

    return tokenized_items
