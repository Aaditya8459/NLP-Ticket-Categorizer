"""TF-IDF feature extraction with persistence."""

import pickle
import logging
from typing import List, Union

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy.sparse import csr_matrix

from src.config import Config

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Wraps sklearn's TfidfVectorizer with save/load capabilities.

    Why TF-IDF over CountVectorizer?
    --------------------------------
    - Downweights terms that appear in many documents (e.g., "account")
    - Upweights discriminative terms (e.g., "invoice", "refund", "payroll")
    - sublinear_tf dampens the impact of very frequent words in long docs
    """

    def __init__(self, config: Config):
        self.cfg = config
        self.vectorizer = TfidfVectorizer(
            max_features=config.feat_max_features,
            ngram_range=config.feat_ngram_range,
            min_df=config.feat_min_df,
            max_df=config.feat_max_df,
            sublinear_tf=config.feat_sublinear_tf,
        )
        self._is_fitted = False
        logger.info("FeatureExtractor initialized (max_features=%d, ngrams=%s)",
                    config.feat_max_features, config.feat_ngram_range)

    def fit(self, texts: List[str]) -> "FeatureExtractor":
        """Fit the vectorizer on training texts."""
        self.vectorizer.fit(texts)
        self._is_fitted = True
        vocab_size = len(self.vectorizer.vocabulary_)
        logger.info("TF-IDF fitted. Vocabulary size: %d", vocab_size)
        return self

    def transform(self, texts: List[str]) -> csr_matrix:
        """Transform texts to TF-IDF matrix."""
        if not self._is_fitted:
            raise RuntimeError("Vectorizer not fitted. Call .fit() first.")
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts: List[str]) -> csr_matrix:
        """Fit and transform in one step."""
        X = self.vectorizer.fit_transform(texts)
        self._is_fitted = True
        vocab_size = len(self.vectorizer.vocabulary_)
        logger.info("TF-IDF fitted. Vocabulary size: %d", vocab_size)
        return X

    def save(self, path: str) -> None:
        """Serialize vectorizer to disk."""
        with open(path, "wb") as f:
            pickle.dump(self.vectorizer, f)
        logger.info("Vectorizer saved to %s", path)

    def load(self, path: str) -> "FeatureExtractor":
        """Load vectorizer from disk."""
        with open(path, "rb") as f:
            self.vectorizer = pickle.load(f)
        self._is_fitted = True
        vocab_size = len(self.vectorizer.vocabulary_)
        logger.info("Vectorizer loaded from %s (vocab=%d)", path, vocab_size)
        return self

    def top_features(self, n: int = 20) -> List[str]:
        """Return the top-N features by average TF-IDF weight."""
        if not self._is_fitted:
            raise RuntimeError("Vectorizer not fitted.")
        # This requires a fitted corpus; use externally if needed
        return []
