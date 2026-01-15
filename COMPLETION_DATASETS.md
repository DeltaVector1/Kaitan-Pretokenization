# Completion/Pretraining Dataset Support

This document describes how to use the pretokenization tool with completion/pretraining datasets.

## Overview

The tool now supports two dataset types:

1. **Chat datasets** (`dan-chat-advanced`): For instruction-following and conversational data with role-based masking
2. **Completion datasets** (`completion`): For pretraining or completion-only data where all tokens are trained on

## Using Completion Datasets

### Basic Configuration

```yaml
base_model: NewEden/Apertus-8B-2509-patched-chatML
tokenizer_type: AutoTokenizer

datasets:
  - path: Delta-Vector/Ursa-Armoured-Core-6-Lora-V1-Kimi
    type: completion
    field: entry  # field containing the text (default: "text")

sequence_len: 8192
```

### Field Parameter

The `field` parameter specifies which field in your dataset contains the text to tokenize:

```yaml
# If your dataset has a "text" field (this is the default)
- path: my-dataset
  type: completion
  # field: text is assumed

# If your dataset has a different field name
- path: my-dataset
  type: completion
  field: entry  # or "content", "document", etc.
```

### Running

```bash
# With interactive confirmation
uv run pretokenization --config ./aperus.yml

# Skip confirmation (useful for scripts/automation)
uv run pretokenization --config ./aperus.yml -y
```

## Mixing Chat and Completion Datasets

You can mix both dataset types in a single configuration:

```yaml
base_model: Qwen/Qwen2.5-1.5B-Instruct
tokenizer_type: AutoTokenizer

datasets:
  # Chat datasets - for instruction tuning
  - path: PocketDoc/Dans-Systemmaxx
    type: dan-chat-advanced
  
  # Completion datasets - for pretraining
  - path: Delta-Vector/Ursa-Armoured-Core-6-Lora-V1-Kimi
    type: completion
    field: entry

sequence_len: 8192
```

When you run with a mixed config:

1. If there are chat datasets, you'll be prompted to select a chat format (or use `--format`)
2. Both chat and completion datasets will be processed
3. All data will be shuffled together in the final output

```bash
# Select format interactively
uv run pretokenization --config ./mixed.yml

# Or specify format on command line
uv run pretokenization --config ./mixed.yml --format "ChatML" -y
```

## How It Works

### Tokenization Behavior

**Chat datasets:**
- Adds special tokens for roles (system, user, assistant)
- Masks tokens based on `loss` field in conversations
- Formats multi-turn dialogues

**Completion datasets:**
- Extracts text from specified field
- Tokenizes with `add_special_tokens=True` (adds BOS token if available)
- **Trains on all tokens** (no masking)
- Truncates to `sequence_len` if needed

### Preview

When processing completion datasets, you'll see a preview like this:

```
================================================================================
COMPLETION DATASET PREVIEW
================================================================================
All tokens are trained on (no masking)
================================================================================
<s>Walter, known to mercenary networks as "Handler Walter," is a survivor...
================================================================================

Total tokens: 831
Unmasked tokens: 831
Masked tokens: 0
================================================================================
```

All tokens are shown in green (unmasked) since completion datasets train on everything.

## Output Format

The output Parquet file has the same schema regardless of dataset type:

```python
{
    "input_ids": List[int],       # Token IDs
    "attention_mask": List[int],  # All 1s (no padding)
    "labels": List[int]           # Token IDs (or -100 for masked)
}
```

For completion datasets:
- `labels` = `input_ids` (no masking)
- All tokens contribute to the loss

For chat datasets:
- `labels` contains -100 for masked tokens
- Only unmasked tokens contribute to the loss

## Example Use Cases

### Pure Pretraining
```yaml
datasets:
  - path: HuggingFaceFW/fineweb
    type: completion
    field: text
  - path: your-org/custom-corpus
    type: completion
    field: document
```

### Pure Instruction Tuning
```yaml
datasets:
  - path: dataset-1
    type: dan-chat-advanced
  - path: dataset-2
    type: dan-chat-advanced
```

### Hybrid Training (Instruction + Pretraining)
```yaml
datasets:
  # Instruction data
  - path: instruction-dataset
    type: dan-chat-advanced
  
  # Domain knowledge
  - path: domain-corpus
    type: completion
    field: text
```

## Tips

1. **Field names**: Common field names are `text`, `content`, `entry`, `document`. Check your dataset structure first.

2. **Sequence length**: Completion samples are truncated to `sequence_len`, so adjust based on your dataset's typical document length.

3. **Memory**: Completion datasets can be large. The tool uses streaming mode to handle datasets of any size.

4. **Mixing ratios**: If mixing datasets, consider the relative sizes. The final shuffle ensures good mixing, but you may want to duplicate smaller datasets in your config.

5. **Skip confirmation**: Use `-y` flag when running in scripts or automated pipelines.

## Testing Your Config

Test with a small dataset first:

```bash
# Enable debug logging to see what's happening
uv run pretokenization --config ./test.yml --debug -y

# Check the output
python -c "
import pyarrow.parquet as pq
table = pq.read_table('output.parquet')
print(f'Rows: {table.num_rows}')
print(f'Sample: {table.to_pylist()[0]}')
"
```

## Troubleshooting

### "Field 'X' not found in item"
Check your dataset structure:
```python
from datasets import load_dataset
ds = load_dataset("your-dataset", split="train", streaming=True)
item = next(iter(ds))
print(item.keys())  # See available fields
```

### Preview shows "Could not generate preview"
The specified field might be empty or missing in the first item. The tool will skip invalid items during processing.

### All tokens are masked (red)
This is normal for chat datasets with `loss: false`. For completion datasets, all tokens should be green (unmasked).
