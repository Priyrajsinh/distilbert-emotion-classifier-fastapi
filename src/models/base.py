"""Abstract base class for all ML models in the B1 project.

Every model implementation (e.g. SentimentClassifier) must inherit from
BaseMLModel and implement all abstract methods defined here.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd


class BaseMLModel(ABC):
    """Abstract interface that all B1 model classes must implement.

    Enforces a consistent contract for fitting, inference, persistence,
    and probability estimation across every model in this project.
    """

    @abstractmethod
    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
        """Train the model on the provided training and validation sets.

        Args:
            train_df: DataFrame with 'text' and 'label' columns for training.
            val_df: DataFrame with 'text' and 'label' columns for validation.
        """

    @abstractmethod
    def predict(self, texts: list[str]) -> list[int]:
        """Run inference and return the predicted class index for each text.

        Args:
            texts: A list of raw input strings.

        Returns:
            A list of integer class indices, one per input text.
        """

    @abstractmethod
    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Return class probability distributions for each input text.

        Args:
            texts: A list of raw input strings.

        Returns:
            A 2-D numpy array of shape (len(texts), num_labels) where each
            row sums to 1.0.
        """

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist the model artefacts to disk at the given path.

        Args:
            path: Directory path where model files will be written.
        """

    @abstractmethod
    def load(self, path: Path) -> "BaseMLModel":
        """Load model artefacts from disk and return the initialised model.

        Args:
            path: Directory path containing previously saved model files.

        Returns:
            The loaded model instance (typically self after mutating state).
        """
