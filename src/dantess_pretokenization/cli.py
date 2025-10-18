"""Main CLI entry point for pretokenization."""

import argparse
import logging
import os
import random
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset
from transformers import AutoTokenizer

from .config import Config
from .formats import FORMATS
from .tokenizer import fill_format_tokens, tokenize_dataset, tokenize_item
from .ui import confirm_tokenization, select_chat_format, show_colored_preview

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Pretokenize HuggingFace datasets with multiple chat formats.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--output", type=str, default="./output.parquet", help="Path to the output parquet file.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--hf-push", type=str, default=None, help="HuggingFace repo to push the dataset to.")
    parser.add_argument("--format", type=str, default=None, help="Chat format to use (skip TUI selection if provided).")

    args = parser.parse_args()

    # Set logging level
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logging.getLogger("dantess_pretokenization").setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled.")
    else:
        logger.setLevel(logging.INFO)

    # Set random seed
    if args.seed:
        random.seed(args.seed)
        logger.debug(f"Random seed set to {args.seed}")

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = Config(args.config)
    logger.info(f"Loaded config: {config}")

    # Get datasets (filter by dan-chat-advanced type)
    dataset_paths = config.get_dataset_paths_by_type("dan-chat-advanced")
    if not dataset_paths:
        logger.error("No datasets with type 'dan-chat-advanced' found in configuration.")
        return

    logger.info(f"Found {len(dataset_paths)} dataset(s) to process: {dataset_paths}")

    # Select chat format
    if args.format:
        if args.format not in FORMATS:
            logger.error(f"Invalid format: {args.format}. Available formats: {list(FORMATS.keys())}")
            return
        selected_format_name = args.format
    else:
        selected_format_name = select_chat_format(list(FORMATS.keys()))

    if not selected_format_name:
        logger.error("No format selected. Exiting.")
        return

    logger.info(f"Selected format: {selected_format_name}")

    # Instantiate format
    format_class = FORMATS[selected_format_name]
    format_obj = format_class()
    format_tokens = format_obj.format_tokens

    # Load tokenizer
    logger.info(f"Loading tokenizer from {config.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=config.trust_remote_code)

    # Fill format tokens
    is_bos_fix = selected_format_name == "ChatML with BOS Fix"
    fill_format_tokens(tokenizer, format_tokens, is_bos_fix)

    # Check that all format tokens are filled
    for key, value in format_tokens.items():
        if value["tokens"] is None:
            raise ValueError(f"Tokenization failed for {key}: {value['text']}")
    logger.info("Format tokens filled successfully.")
    logger.debug(f"Format tokens: {format_tokens}")

    # Preview tokenization with one example
    logger.info("Loading one example for preview...")
    preview_dataset_path = dataset_paths[0]
    try:
        from datasets import load_dataset

        preview_stream = load_dataset(preview_dataset_path, split="train", streaming=True, trust_remote_code=True)
        preview_item = next(iter(preview_stream))
    except Exception:
        from datasets import load_dataset

        preview_stream = load_dataset(preview_dataset_path, split="train", streaming=True)
        preview_item = next(iter(preview_stream))

    preview_tokenized = tokenize_item(preview_item, tokenizer, format_tokens, format_obj, config.sequence_len)

    if preview_tokenized:
        show_colored_preview(
            preview_tokenized["input_ids"], preview_tokenized["labels"], tokenizer, format_obj.mask_token_id
        )
    else:
        logger.warning("Could not generate preview (item was filtered out).")

    # Confirm before proceeding
    if not confirm_tokenization():
        logger.info("Tokenization cancelled by user.")
        return

    # Create temporary directory for tokenized data
    temp_dir = tempfile.TemporaryDirectory()
    logger.debug(f"Temporary directory created at {temp_dir.name}")

    # Tokenize each dataset
    all_tokenized_data = []
    for dataset_path in dataset_paths:
        logger.info(f"Tokenizing dataset: {dataset_path}")
        tokenized_data = tokenize_dataset(dataset_path, tokenizer, format_tokens, format_obj, config.sequence_len)
        logger.info(f"Tokenized {len(tokenized_data)} items from {dataset_path}.")

        # Log statistics
        total_tokens = sum(len(item["input_ids"]) for item in tokenized_data)
        logger.info(f"Total tokens in {dataset_path}: {total_tokens}")

        total_unmasked_tokens = sum(
            len(item["input_ids"]) - item["labels"].count(format_obj.mask_token_id) for item in tokenized_data
        )
        logger.info(f"Total unmasked tokens in {dataset_path}: {total_unmasked_tokens}")

        # Save to temporary parquet file
        temp_file = os.path.join(temp_dir.name, f"{Path(dataset_path).name}.parquet")
        table = pa.Table.from_pylist(tokenized_data)
        del tokenized_data
        pq.write_table(table, temp_file, compression="snappy")
        logger.info(f"Tokenized data saved to {temp_file}.")

    # Combine all tokenized data
    logger.info("Combining all tokenized datasets...")
    for dataset_path in dataset_paths:
        temp_file = os.path.join(temp_dir.name, f"{Path(dataset_path).name}.parquet")
        table = pq.read_table(temp_file)
        all_tokenized_data.extend(table.to_pylist())
        logger.info(f"Loaded tokenized data from {temp_file}.")

    # Shuffle
    logger.info("Shuffling combined dataset...")
    random.shuffle(all_tokenized_data)

    # Log final statistics
    total_tokens = sum(len(item["input_ids"]) for item in all_tokenized_data)
    logger.info(f"Total tokens in combined dataset: {total_tokens}")

    total_unmasked_tokens = sum(
        len(item["input_ids"]) - item["labels"].count(format_obj.mask_token_id) for item in all_tokenized_data
    )
    logger.info(f"Total unmasked tokens in combined dataset: {total_unmasked_tokens}")

    # Save to output file
    output_file = args.output
    logger.info(f"Saving tokenized data to {output_file}...")

    table = pa.Table.from_pylist(all_tokenized_data)
    del all_tokenized_data
    pq.write_table(table, output_file, compression="snappy")
    del table

    logger.info(f"Tokenized data saved to {output_file}.")

    # Push to HuggingFace if requested
    if args.hf_push:
        logger.info(f"Pushing dataset to HuggingFace: {args.hf_push}")
        dataset = Dataset.from_parquet(output_file)
        dataset.push_to_hub(args.hf_push, private=True, token=os.getenv("HF_TOKEN"), max_shard_size="5GB")
        logger.info(f"Dataset pushed to HuggingFace: {args.hf_push}")

    # Cleanup
    temp_dir.cleanup()
    logger.info("Done!")


if __name__ == "__main__":
    main()
