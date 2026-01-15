
<img src="https://files.catbox.moe/uja15m.png" alt="description" width="200" height="200">

# Kaitan Pretokenization

Simple, extensible pretokenization project for HF datasets with multiple chat formats

## Install

Install UV:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Activate a venv:
```bash
uv venv
. .venv/bin/activate
```
and then install deps:
```bash
uv sync
```

## Usage

### Basic Usage

```bash
uv run pretokenization --config ./testing.yml
```

This will:
1. Load your configuration
2. Show a TUI to select the chat format
3. Display a colored preview of tokenization
4. Ask for confirmation
5. Process all datasets and save to `output.parquet`

### Yaml File

Create a YAML configuration file

```yaml
base_model: Qwen/Qwen2.5-1.5B-Instruct
model_type: AutoModelForCausalLM
tokenizer_type: AutoTokenizer

datasets:
  - path: PocketDoc/Dans-Systemmaxx
    type: dan-chat-advanced
  - path: Delta-Vector/Hydrus-Claude-Instruct-2.7K
    type: dan-chat-advanced
  - path: Delta-Vector/Ursa-Armoured-Core-6-Lora-V1-Kimi
    type: completion
    field: entry  # field name to extract text from (default: "text")

sequence_len: 8192  # defaults to 8192
```

### Command Line Options

```bash
uv run pretokenization --config CONFIG [OPTIONS]

Options:
  --config CONFIG      Path to the YAML configuration file (required)
  --output OUTPUT      Path to output parquet file (default: ./output.parquet)
  --format FORMAT      Skip TUI and use specific format (for chat datasets)
  --seed SEED          Random seed for reproducibility
  --debug              Enable debug logging
  --hf-push REPO       Push results to HuggingFace repo
  -y, --yes            Skip confirmation prompt
```

### Examples

```bash
# Basic usage with TUI
uv run pretokenization --config ./testing.yml

# Skip TUI, use ChatML format
uv run pretokenization --config ./testing.yml --format "ChatML"

# Save to custom location
uv run pretokenization --config ./testing.yml --output ./my_dataset.parquet

# With debug logging
uv run pretokenization --config ./testing.yml --debug

# Push to HuggingFace (requires HF_TOKEN env var)
uv run pretokenization --config ./testing.yml --hf-push username/dataset-name

# Process completion dataset (pretraining)
uv run pretokenization --config ./aperus.yml -y

# Mix chat and completion datasets
uv run pretokenization --config ./mixed.yml --format "ChatML" -y
```

## Adding New Chat Formats

To add a new chat format, create a new file in `src/dantess_pretokenization/formats/`:

```python
# src/dantess_pretokenization/formats/my_format.py
from .base import BaseChatFormat

class MyFormat(BaseChatFormat):
    @property
    def name(self) -> str:
        return "My Custom Format"
    
    @property
    def format_tokens(self) -> dict:
        return {
            "starting_sequence": {"text": "<start>", "tokens": None},
            "system_prefix": {"text": "<sys>", "tokens": None},
            "user_prefix": {"text": "<user>", "tokens": None},
            "assistant_prefix": {"text": "<assistant>", "tokens": None},
            "tool_prefix": {"text": "<tool>", "tokens": None},
            "turn_seperator": {"text": "\n", "tokens": None},
            "turn_suffix": {"text": "</turn>", "tokens": None},
        }
```

Then register it in `src/dantess_pretokenization/formats/__init__.py`:

```python
from .my_format import MyFormat

FORMATS = {
    # ... existing formats ...
    "My Custom Format": MyFormat,
}
```

## Dataset Formats

The tool supports two dataset types:

### 1. Chat Format (`dan-chat-advanced`)

For instruction/conversation datasets:

```json
{
  "conversations": [
    {
      "from": "system",
      "value": "System message content"
    },
    {
      "from": "user",
      "value": "User message",
      "loss": false
    },
    {
      "from": "assistant",
      "value": "Assistant response",
      "loss": true,
      "prefix": "Optional prefix (not trained)"
    }
  ]
}
```

### 2. Completion Format (`completion`)

For pretraining/completion datasets where all tokens are trained on:

```yaml
datasets:
  - path: your-username/your-dataset
    type: completion
    field: text  # field name containing the text (default: "text")
```

The completion format will tokenize the specified field and train on all tokens (no masking). This is useful for:
- Pretraining datasets
- Completion-only datasets
- Knowledge/documentation corpora

## Output Format

The tool outputs a Parquet file with the following schema:

```python
{
  "input_ids": List[int],      # Token IDs
  "attention_mask": List[int], # Attention mask (all 1s)
  "labels": List[int]          # Labels (-100 for masked tokens)
}
```

## Development

```bash
# Install dev dependencies
uv add --dev ruff

# Run linter
uv run ruff check src/

# Format code
uv run ruff format src/

# Fix issues automatically
uv run ruff check src/ --fix
```

