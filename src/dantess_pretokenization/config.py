"""Configuration loading and validation."""

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


class Config:
    """Configuration container for pretokenization."""

    def __init__(self, config_path: str | Path) -> None:
        """Load configuration from YAML file.

        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = Path(config_path)
        with open(self.config_path) as file:
            self._data = yaml.safe_load(file)

    @property
    def base_model(self) -> str:
        """Get the base model path."""
        return self._data.get("base_model")

    @property
    def model_type(self) -> str:
        """Get the model type."""
        return self._data.get("model_type", "AutoModelForCausalLM")

    @property
    def tokenizer_type(self) -> str:
        """Get the tokenizer type."""
        return self._data.get("tokenizer_type", "AutoTokenizer")

    @property
    def sequence_len(self) -> int:
        """Get the sequence length."""
        return self._data.get("sequence_len", 8192)

    @property
    def datasets(self) -> list[dict[str, str]]:
        """Get the list of datasets."""
        return self._data.get("datasets", [])

    @property
    def trust_remote_code(self) -> bool:
        """Get trust_remote_code setting."""
        return self._data.get("trust_remote_code", False)

    def get_dataset_paths_by_type(self, dataset_type: str) -> list[str]:
        """Get dataset paths filtered by type.

        Args:
            dataset_type: The dataset type to filter by

        Returns:
            List of dataset paths
        """
        return [d["path"] for d in self.datasets if d.get("type") == dataset_type]
    
    def get_datasets_by_type(self, dataset_type: str) -> list[dict]:
        """Get full dataset configs filtered by type.

        Args:
            dataset_type: The dataset type to filter by

        Returns:
            List of dataset configurations
        """
        return [d for d in self.datasets if d.get("type") == dataset_type]

    def __repr__(self) -> str:
        return f"Config(base_model={self.base_model}, datasets={len(self.datasets)})"
