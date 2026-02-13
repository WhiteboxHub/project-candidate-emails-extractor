import logging
import os
from typing import Optional

import joblib

logger = logging.getLogger(__name__)


class MLFilter:
    """Encapsulates ML recruiter-vs-non-recruiter classification."""

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.classifier = None
        self.vectorizer = None

    def load(self) -> bool:
        classifier_path = os.path.join(self.model_dir, "classifier.pkl")
        vectorizer_path = os.path.join(self.model_dir, "vectorizer.pkl")

        if not (os.path.exists(classifier_path) and os.path.exists(vectorizer_path)):
            logger.warning("ML model files not found in %s", self.model_dir)
            return False

        try:
            self.classifier = joblib.load(classifier_path)
            self.vectorizer = joblib.load(vectorizer_path)
            logger.info("ML classifier loaded from %s", self.model_dir)
            return True
        except Exception as error:
            logger.error("Failed to load ML model: %s", error)
            self.classifier = None
            self.vectorizer = None
            return False

    def predict_recruiter(self, subject: str, body: str, from_email: str) -> Optional[bool]:
        """Returns True/False if prediction succeeds, None on inference failure."""
        if not self.classifier or not self.vectorizer:
            return None

        try:
            feature_text = f"{subject or ''} {body or ''} {from_email or ''}"
            features = self.vectorizer.transform([feature_text])
            prediction = self.classifier.predict(features)[0]
            return bool(int(prediction) == 1)
        except Exception as error:
            logger.error("ML classification failed: %s", error)
            return None
