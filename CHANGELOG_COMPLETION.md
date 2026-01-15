# Changelog: Completion Dataset Support

## Changes Made

### New Features

1. **Completion Dataset Support**
   - Added new `completion` dataset type for pretraining/completion datasets
   - All tokens are trained on (no masking)
   - Configurable field extraction via `field` parameter (default: "text")

2. **Field-based Text Extraction**
   - Users can specify which field contains the text to tokenize
   - Supports any field name: `text`, `entry`, `content`, `document`, etc.

3. **Mixed Dataset Processing**
   - Can process both chat (`dan-chat-advanced`) and completion datasets in one run
   - Final output is shuffled together

4. **Skip Confirmation Flag**
   - Added `-y` / `--yes` flag to skip interactive confirmation
   - Useful for automation and scripts

### Files Modified

#### `src/dantess_pretokenization/tokenizer.py`
- Added `tokenize_completion_item()` function
- Added `tokenize_completion_dataset()` function
- Handles text extraction, tokenization with special tokens, and no masking

#### `src/dantess_pretokenization/config.py`
- Added `get_datasets_by_type()` method
- Returns full dataset configurations (including field parameter)

#### `src/dantess_pretokenization/cli.py`
- Added support for processing completion datasets
- Added completion dataset preview
- Added `-y` / `--yes` flag for skipping confirmation
- Updated logic to handle both dataset types in single run
- Improved statistics logging for mixed datasets

#### Documentation Updates
- `README.md`: Added completion dataset examples and documentation
- `QUICKSTART.md`: Added completion dataset quick start guide
- `COMPLETION_DATASETS.md`: Comprehensive guide for completion datasets

### Files Added

1. **aperus.yml**: Test configuration for completion datasets
2. **mixed_example.yml**: Example mixing both chat and completion datasets
3. **COMPLETION_DATASETS.md**: Complete documentation for completion datasets
4. **CHANGELOG_COMPLETION.md**: This changelog

### Example Configurations

#### Completion Only
```yaml
base_model: NewEden/Apertus-8B-2509-patched-chatML
datasets:
  - path: Delta-Vector/Ursa-Armoured-Core-6-Lora-V1-Kimi
    type: completion
    field: entry
```

#### Mixed Chat + Completion
```yaml
base_model: Qwen/Qwen2.5-1.5B-Instruct
datasets:
  - path: PocketDoc/Dans-Systemmaxx
    type: dan-chat-advanced
  - path: Delta-Vector/Ursa-Armoured-Core-6-Lora-V1-Kimi
    type: completion
    field: entry
```

### Usage Examples

```bash
# Process completion dataset
uv run pretokenization --config ./aperus.yml -y

# Mix chat and completion (specify format for chat)
uv run pretokenization --config ./mixed.yml --format "ChatML" -y

# With debug logging
uv run pretokenization --config ./aperus.yml --debug -y
```

### Testing

Tested with:
- **Model**: `NewEden/Apertus-8B-2509-patched-chatML`
- **Dataset**: `Delta-Vector/Ursa-Armoured-Core-6-Lora-V1-Kimi`
- **Field**: `entry`
- **Results**: 
  - 166 items processed
  - 94,723 total tokens
  - All tokens unmasked (trained on)
  - Successfully saved to Parquet format

### Backward Compatibility

✅ All existing functionality preserved:
- Chat datasets work exactly as before
- Existing configs continue to work
- No breaking changes

### Key Implementation Details

1. **Tokenization**: Uses `tokenizer.encode()` with `add_special_tokens=True`
2. **Masking**: Completion datasets have `labels = input_ids` (no -100 masking)
3. **Truncation**: Sequences truncated to `sequence_len` if needed
4. **Streaming**: Uses HuggingFace streaming for memory efficiency
5. **Concurrency**: ThreadPoolExecutor for parallel processing

### Output Format

Same schema for all dataset types:
```python
{
    "input_ids": List[int],       # Token IDs
    "attention_mask": List[int],  # All 1s
    "labels": List[int]           # Token IDs or -100 for masked
}
```

For completion datasets: `labels = input_ids` (no masking)

## Benefits

1. **Unified Tool**: Single tool for both instruction tuning and pretraining
2. **Flexible**: Support for any field name in any dataset structure
3. **Efficient**: Streaming mode handles large datasets
4. **Simple**: Easy configuration with YAML
5. **Compatible**: Works with existing chat dataset workflows
