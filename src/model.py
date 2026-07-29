"""Classifier training, evaluation, and persistence."""

import pickle
import logging
from typing import Tuple, Dict, Any, Optional

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from src.config import Config

logger = logging.getLogger(__name__)


class TicketClassifier:
    """
    Wrapper around sklearn classifiers with unified interface.

    Supports:
        - Multinomial Naive Bayes (fast, probabilistic, good baseline)
        - Logistic Regression (calibrated probabilities, L2 regularization)
    """

    def __init__(self, config: Config):
        self.cfg = config
        self.model = None
        self.classes_: Optional[np.ndarray] = None
        self._is_trained = False

    def build(self) -> "TicketClassifier":
        """Instantiate the underlying sklearn model."""
        if self.cfg.model_type == "naive_bayes":
            self.model = MultinomialNB(alpha=self.cfg.nb_alpha)
            logger.info("Built MultinomialNB (alpha=%.2f)", self.cfg.nb_alpha)
        elif self.cfg.model_type == "logistic_regression":
            self.model = LogisticRegression(
                C=self.cfg.lr_c,
                max_iter=self.cfg.lr_max_iter,
                random_state=self.cfg.random_state,
                n_jobs=-1,
            )
            logger.info("Built LogisticRegression (C=%.2f, max_iter=%d)",
                        self.cfg.lr_c, self.cfg.lr_max_iter)
        else:
            raise ValueError(f"Unknown model type: {self.cfg.model_type}")
        return self

    def train(self, X: csr_matrix, y: np.ndarray) -> "TicketClassifier":
        """Train the classifier."""
        if self.model is None:
            self.build()
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        self._is_trained = True
        logger.info("Model trained on %d samples, %d classes", X.shape[0], len(self.classes_))
        return self

    def predict(self, X: csr_matrix) -> np.ndarray:
        """Predict class labels."""
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call .train() or .load() first.")
        return self.model.predict(X)

    def predict_proba(self, X: csr_matrix) -> np.ndarray:
        """Predict class probabilities."""
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call .train() or .load() first.")
        return self.model.predict_proba(X)

    def evaluate(self, X: csr_matrix, y_true: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate model performance.

        Returns dict with:
            - accuracy
            - precision, recall, f1 (weighted)
            - per-class report
            - confusion matrix
        """
        y_pred = self.predict(X)
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )

        report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
        cm = confusion_matrix(y_true, y_pred, labels=self.classes_)

        metrics = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "per_class": report,
            "confusion_matrix": cm.tolist(),
            "classes": self.classes_.tolist(),
        }

        logger.info("Evaluation — Accuracy: %.3f | F1: %.3f", acc, f1)
        return metrics

    def save(self, path: str) -> None:
        """Serialize model to disk."""
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        logger.info("Model saved to %s", path)

    def load(self, path: str) -> "TicketClassifier":
        """Load model from disk."""
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        self.classes_ = self.model.classes_
        self._is_trained = True
        logger.info("Model loaded from %s", path)
        return self
