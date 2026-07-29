"""Main orchestrator: preprocess → vectorize → predict → route → prioritize."""

import logging
from typing import Dict, List, Any
from dataclasses import dataclass, asdict

import numpy as np
from scipy.sparse import csr_matrix

from src.config import Config
from src.preprocessor import TextPreprocessor
from src.features import FeatureExtractor
from src.model import TicketClassifier

logger = logging.getLogger(__name__)


@dataclass
class TriageResult:
    """
    Structured output for a single ticket prediction.

    Attributes:
        subject: Original ticket subject.
        body: Original ticket body.
        predicted_category: Best-guess category.
        confidence: Probability of the predicted class (0.0–1.0).
        confidence_pct: Human-readable confidence string.
        all_probabilities: Dict mapping every class to its probability.
        priority: "URGENT" or "NORMAL" based on keyword rules.
        routing: "AUTO-ASSIGNED" or "NEEDS HUMAN REVIEW".
        routing_reason: Explanation for the routing decision.
        cleaned_text: The preprocessed text (for debugging).
    """
    subject: str
    body: str
    predicted_category: str
    confidence: float
    confidence_pct: str
    all_probabilities: Dict[str, float]
    priority: str
    routing: str
    routing_reason: str
    cleaned_text: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to plain dict for JSON serialization."""
        return asdict(self)

    def to_markdown(self) -> str:
        """Pretty-print for CLI/dashboard display."""
        lines = [
            f"📧 **{self.subject}**",
            f"   🏷️  Category: `{self.predicted_category}`",
            f"   📊 Confidence: `{self.confidence_pct}`",
            f"   🚨 Priority:   `{self.priority}`",
            f"   🔄 Routing:    `{self.routing}`",
            f"   📝 Reason:     {self.routing_reason}",
            f"   📈 Probs:      {self.all_probabilities}",
        ]
        return "\n".join(lines)


class TriageEngine:
    """
    End-to-end ticket triage system.

    Usage:
        engine = TriageEngine(config)
        engine.load_artifacts()   # after training
        result = engine.predict_one(subject, body)
    """

    def __init__(self, config: Config):
        self.cfg = config
        self.preprocessor = TextPreprocessor(config)
        self.extractor = FeatureExtractor(config)
        self.classifier = TicketClassifier(config)
        logger.info("TriageEngine initialized")

    def train(self, subjects: List[str], bodies: List[str], labels: List[str]) -> Dict[str, Any]:
        """
        Full training pipeline.

        Args:
            subjects: List of ticket subjects.
            bodies: List of ticket bodies.
            labels: List of category labels.

        Returns:
            Evaluation metrics dict.
        """
        # 1. Combine and clean
        raw_texts = [
            f"{s}{self.cfg.data_text_separator}{b}"
            for s, b in zip(subjects, bodies)
        ]
        logger.info("Cleaning %d tickets...", len(raw_texts))
        cleaned = self.preprocessor.clean_batch(raw_texts)

        # 2. Vectorize
        logger.info("Vectorizing...")
        X = self.extractor.fit_transform(cleaned)
        y = np.array(labels)

        # 3. Train/test split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.cfg.test_size,
            random_state=self.cfg.random_state,
            stratify=y,
        )
        logger.info("Split: %d train | %d test", X_train.shape[0], X_test.shape[0])

        # 4. Train model
        self.classifier.build().train(X_train, y_train)

        # 5. Evaluate
        metrics = self.classifier.evaluate(X_test, y_test)

        # 6. Save artifacts
        self._save_artifacts()

        return metrics

    def predict_one(self, subject: str, body: str) -> TriageResult:
        """
        Predict a single ticket with full routing logic.

        Args:
            subject: Ticket subject line.
            body: Ticket body text.

        Returns:
            TriageResult with category, confidence, priority, and routing.
        """
        # Preprocess
        raw = f"{subject}{self.cfg.data_text_separator}{body}"
        cleaned = self.preprocessor.clean(raw)

        # Vectorize
        X = self.extractor.transform([cleaned])

        # Predict probabilities
        proba = self.classifier.predict_proba(X)[0]
        pred_idx = int(np.argmax(proba))
        pred_label = self.classifier.classes_[pred_idx]
        confidence = float(proba[pred_idx])

        # All probabilities
        all_probs = {
            cls: float(p) for cls, p in zip(self.classifier.classes_, proba)
        }

        # Priority
        priority = self._detect_priority(subject, body)

        # Routing
        if confidence < self.cfg.confidence_threshold:
            routing = "NEEDS HUMAN REVIEW"
            reason = (f"Confidence {confidence:.1%} is below the "
                      f"{self.cfg.confidence_threshold:.0%} auto-assignment threshold")
        else:
            routing = "AUTO-ASSIGNED"
            reason = f"Confidence {confidence:.1%} meets threshold"

        return TriageResult(
            subject=subject,
            body=body,
            predicted_category=pred_label,
            confidence=confidence,
            confidence_pct=f"{confidence:.1%}",
            all_probabilities=all_probs,
            priority=priority,
            routing=routing,
            routing_reason=reason,
            cleaned_text=cleaned[:200] + "..." if len(cleaned) > 200 else cleaned,
        )

    def predict_batch(self, subjects: List[str], bodies: List[str]) -> List[TriageResult]:
        """Predict multiple tickets."""
        return [self.predict_one(s, b) for s, b in zip(subjects, bodies)]

    def _detect_priority(self, subject: str, body: str) -> str:
        """Rule-based urgency detection."""
        combined = (subject + " " + body).lower()
        hits = sum(1 for kw in self.cfg.urgent_keywords if kw in combined)
        return "URGENT" if hits >= self.cfg.urgent_threshold else "NORMAL"

    def _save_artifacts(self) -> None:
        """Persist all trained components."""
        self.extractor.save(self.cfg.vectorizer_file)
        self.classifier.save(self.cfg.model_file)
        logger.info("All artifacts saved.")

    def load_artifacts(self) -> "TriageEngine":
        """Load previously trained components from disk."""
        self.extractor.load(self.cfg.vectorizer_file)
        self.classifier.load(self.cfg.model_file)
        logger.info("All artifacts loaded.")
        return self
