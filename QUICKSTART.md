# Quick Start Guide

## Installation

```bash
# Clone/navigate to the project directory
cd Dantess-Pretokenization

# Install dependencies with UV
uv sync
```

## Basic Usage

### 1. Run with Interactive TUI (Recommended)

```bash
uv run pretokenization --config ./testing.yml
```

This will:
1. ✅ Load your configuration
2. ✅ Show a menu to select chat format
3. ✅ Display a colored preview of how tokens will be masked
4. ✅ Ask for confirmation
5. ✅ Process all datasets and save to `output.parquet`

### 2. Skip TUI (Direct Format Selection)

```bash
uv run pretokenization --config ./testing.yml --format "ChatML"
```

Available formats:
- `ChatML`
- `ChatML with BOS Fix`
- `GLM4`
- `Llama3`
- `Llama4`
- `Mistral V7 Tekken`

### 3. Custom Output Location

```bash
uv run pretokenization --config ./testing.yml --output ./my_tokenized_data.parquet
```

## Configuration File

The `testing.yml` file defines what to tokenize:

```yaml
base_model: Qwen/Qwen2.5-1.5B-Instruct  # Model for tokenizer
model_type: AutoModelForCausalLM         # Model type (optional)
tokenizer_type: AutoTokenizer            # Tokenizer type (optional)

datasets:
  - path: PocketDoc/Dans-Systemmaxx               # HuggingFace dataset path
    type: dan-chat-advanced                        # Chat dataset format
  - path: Delta-Vector/Hydrus-Claude-Instruct-2.7K
    type: dan-chat-advanced
  - path: Delta-Vector/Ursa-Armoured-Core-6-Lora-V1-Kimi
    type: completion                               # Pretraining/completion format
    field: entry                                   # Field to extract text from

sequence_len: 8192  # Max sequence length (optional, default: 8192)
```

## Dataset Types

### Chat Datasets (`dan-chat-advanced`)
For instruction-following and conversational datasets. You'll select a chat format (ChatML, Llama3, etc.) and the tool will:
- Add special tokens for roles (system, user, assistant)
- Mask tokens based on the `loss` field in conversations
- Format multi-turn dialogues

### Completion Datasets (`completion`)
For pretraining or completion-only datasets. The tool will:
- Extract text from the specified field (default: `text`)
- Train on **all tokens** (no masking)
- Add BOS token if the tokenizer has one

**Example config for completion:**
```yaml
datasets:
  - path: your-dataset/path
    type: completion
    field: entry  # or "text", "content", etc.
```

## Understanding the Preview

When you run the tool, you'll see a colored preview like this:

```
================================================================================
TOKENIZATION PREVIEW
================================================================================
Green = Unmasked (trained)
Red = Masked (not trained)
================================================================================
[RED]<|im_start|>system[/RED]
[RED]You are a helpful assistant.[/RED]
[RED]<|im_end|>[/RED]
[RED]<|im_start|>user[/RED]
[RED]Hello![/RED]
[RED]<|im_end|>[/RED]
[RED]<|im_start|>assistant[/RED]
[GREEN]Hi there! How can I help you today?[/GREEN]
[RED]<|im_end|>[/RED]
================================================================================
Total tokens: 45
Unmasked tokens: 12
Masked tokens: 33
================================================================================
```

- **Green tokens**: The model will be trained on these
- **Red tokens**: These are masked (loss = False)

## Common Commands

```bash
# Show help
uv run pretokenization --help

# Basic usage with TUI
uv run pretokenization --config ./testing.yml

# Use specific format (skip TUI)
uv run pretokenization --config ./testing.yml --format "GLM4"

# Save to custom location
uv run pretokenization --config ./testing.yml --output ./data.parquet

# Enable debug logging
uv run pretokenization --config ./testing.yml --debug

# Set random seed for reproducibility
uv run pretokenization --config ./testing.yml --seed 42

# Push to HuggingFace (requires HF_TOKEN environment variable)
export HF_TOKEN=your_token_here
uv run pretokenization --config ./testing.yml --hf-push username/dataset-name

# Process completion/pretraining datasets (skip confirmation)
uv run pretokenization --config ./aperus.yml -y
```

## Output Format

The tool creates a Parquet file with:

```python
{
    "input_ids": [1, 2, 3, ...],        # Token IDs
    "attention_mask": [1, 1, 1, ...],   # Attention mask (all 1s)
    "labels": [1, -100, 3, ...]         # Labels (-100 = masked)
}
```

Load it with:

```python
from datasets import load_dataset

dataset = load_dataset("parquet", data_files="output.parquet")
print(dataset["train"][0])
```

## Troubleshooting

### "No datasets with type 'dan-chat-advanced' or 'completion' found"
Make sure your YAML has datasets with `type: dan-chat-advanced` (for chat) or `type: completion` (for pretraining)

### "Failed to load dataset"
- Check the dataset path is correct
- Verify you have internet connection
- Try adding `trust_remote_code: true` to your YAML

### Preview shows all red tokens
This is normal if your dataset has `loss: false` for all assistant responses. Check your data format.

## Adding a New Chat Format

See `README.md` for detailed instructions on adding new formats. It's as simple as:

1. Create a new file in `src/dantess_pretokenization/formats/`
2. Define the format tokens
3. Register it in `__init__.py`

Done! It will automatically appear in the TUI.

## Development

```bash
# Run linter
uv run ruff check src/

# Format code
uv run ruff format src/

# Fix issues automatically
uv run ruff check src/ --fix
```

## Need Help?

- Check `README.md` for full documentation
- See `PROJECT_SUMMARY.md` for architecture details
- Look at `Ref-Scripts/` for original implementation examples
