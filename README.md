# Dantess-Pretokenization

Simple, extensible pretokenization for HuggingFace datasets with multiple chat formats.

## Features

- 🚀 **Multiple Chat Formats**: Built-in support for ChatML, GLM4, Llama3, Llama4, and Mistral V7 Tekken
- 🎨 **Interactive TUI**: Beautiful terminal UI for format selection
- 👀 **Preview Mode**: See colored preview of tokenization before processing
- 📦 **Streaming Support**: Process large datasets efficiently with streaming
- 🔧 **Easily Extensible**: Add new chat formats by creating simple configuration classes
- ⚡ **Concurrent Processing**: Fast tokenization with ThreadPoolExecutor

## Installation

This project uses UV for dependency management:

```bash
# Install dependencies
uv sync

# Or with pip (after building)
pip install -e .
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

### Configuration File

Create a YAML configuration file (e.g., `testing.yml`):

```yaml
base_model: Qwen/Qwen2.5-1.5B-Instruct
model_type: AutoModelForCausalLM
tokenizer_type: AutoTokenizer

datasets:
  - path: PocketDoc/Dans-Systemmaxx
    type: dan-chat-advanced
  - path: Delta-Vector/Hydrus-Claude-Instruct-2.7K
    type: dan-chat-advanced

sequence_len: 8192  # Optional, defaults to 8192
```

### Command Line Options

```bash
uv run pretokenization --config CONFIG [OPTIONS]

Options:
  --config CONFIG      Path to the YAML configuration file (required)
  --output OUTPUT      Path to output parquet file (default: ./output.parquet)
  --format FORMAT      Skip TUI and use specific format
  --seed SEED          Random seed for reproducibility
  --debug              Enable debug logging
  --hf-push REPO       Push results to HuggingFace repo
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
```

## Available Chat Formats

- **ChatML**: Standard ChatML format (`<|im_start|>`, `<|im_end|>`)
- **ChatML with BOS Fix**: ChatML with proper BOS token handling
- **GLM4**: GLM4 chat format with special tokens
- **Llama3**: Llama3 format with header tags
- **Llama4**: Llama4 format with updated header tags
- **Mistral V7 Tekken**: Mistral V7 Tekken format with instruction tags

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

## Dataset Format

The tool expects datasets in the "dan-chat-advanced" format:

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

## License

MIT
