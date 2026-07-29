import os
import yaml
from typing import Any, Dict, List
from pathlib import Path


class Config:
    """Typed configuration container loaded from YAML."""

    def __init__(self, config_path: str = "config.yaml"):
        self._config_path = config_path
        self._raw = self._load()

        self.data_train_path: str = self._raw["data"]["train_path"]
        self.data_subject_col: str = self._raw["data"]["subject_col"]
        self.data_body_col: str = self._raw["data"]["body_col"]
        self.data_label_col: str = self._raw["data"]["label_col"]
        self.data_text_separator: str = self._raw["data"]["text_separator"]

        self.prep_lowercase: bool = self._raw["preprocessing"]["lowercase"]
        self.prep_remove_urls: bool = self._raw["preprocessing"]["remove_urls"]
        self.prep_remove_emails: bool = self._raw["preprocessing"]["remove_emails"]
        self.prep_remove_phones: bool = self._raw["preprocessing"]["remove_phone_numbers"]
        self.prep_remove_numbers: bool = self._raw["preprocessing"]["remove_numbers"]
        self.prep_remove_punct: bool = self._raw["preprocessing"]["remove_punctuation"]
        self.prep_min_token_len: int = self._raw["preprocessing"]["min_token_length"]
        self.prep_stopwords_src: str = self._raw["preprocessing"]["stopwords_source"]
        self.prep_stemmer: str = self._raw["preprocessing"]["stemmer"]

        self.feat_max_features: int = self._raw["features"]["max_features"]
        self.feat_ngram_range: tuple = tuple(self._raw["features"]["ngram_range"])
        self.feat_min_df: int = self._raw["features"]["min_df"]
        self.feat_max_df: float = self._raw["features"]["max_df"]
        self.feat_sublinear_tf: bool = self._raw["features"]["sublinear_tf"]

        self.model_type: str = self._raw["model"]["type"]
        self.lr_c: float = self._raw["model"]["lr_c"]
        self.lr_max_iter: int = self._raw["model"]["lr_max_iter"]
        self.nb_alpha: float = self._raw["model"]["nb_alpha"]
        self.test_size: float = self._raw["model"]["test_size"]
        self.random_state: int = self._raw["model"]["random_state"]

        self.confidence_threshold: float = self._raw["triage"]["confidence_threshold"]
        self.urgent_keywords: List[str] = self._raw["triage"]["urgent_keywords"]
        self.urgent_threshold: int = self._raw["triage"]["urgent_threshold"]

        self.model_dir: str = self._raw["output"]["model_dir"]
        self.vectorizer_file: str = self._raw["output"]["vectorizer_file"]
        self.model_file: str = self._raw["output"]["model_file"]
        self.label_encoder_file: str = self._raw["output"]["label_encoder_file"]
        self.reports_dir: str = self._raw["output"]["reports_dir"]

        self._ensure_dirs()

    def _load(self) -> Dict[str, Any]:
        """Load and parse YAML config."""
        if not os.path.exists(self._config_path):
            raise FileNotFoundError(f"Config file not found: {self._config_path}")
        with open(self._config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _ensure_dirs(self) -> None:
        """Create output directories if they don't exist."""
        for d in [self.model_dir, self.reports_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        return f"Config(path={self._config_path}, model={self.model_type})"
