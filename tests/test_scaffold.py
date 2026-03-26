"""Day 0 scaffold smoke tests.

These stubs verify that the core module contracts are in place before
any training code is written.  All tests must pass on Day 0.
"""

import logging

import pytest


def test_emotion_schema_exists():
    """EMOTION_SCHEMA must be importable and truthy from validation module."""
    from src.data.validation import EMOTION_SCHEMA

    assert EMOTION_SCHEMA


def test_pydantic_input_rejects_empty():
    """SentimentInput must raise validation error for empty text."""
    from src.data.schemas import SentimentInput

    with pytest.raises(Exception):
        SentimentInput(text="")


def test_pydantic_input_rejects_whitespace():
    """SentimentInput must reject whitespace-only strings."""
    from src.data.schemas import SentimentInput

    with pytest.raises(Exception):
        SentimentInput(text="   ")


def test_pydantic_input_accepts_valid_text():
    """SentimentInput must accept a non-empty text string."""
    from src.data.schemas import SentimentInput

    obj = SentimentInput(text="I am feeling great today")
    assert obj.text == "I am feeling great today"


def test_exceptions_hierarchy():
    """DataLoadError must be a subclass of ProjectBaseError."""
    from src.exceptions import DataLoadError, ProjectBaseError

    assert issubclass(DataLoadError, ProjectBaseError)


def test_all_exceptions_inherit_base():
    """All custom exceptions must inherit from ProjectBaseError."""
    from src.exceptions import (
        ConfigError,
        DataLoadError,
        ModelNotFoundError,
        PredictionError,
        ProjectBaseError,
    )

    for exc_cls in (DataLoadError, ModelNotFoundError, PredictionError, ConfigError):
        assert issubclass(exc_cls, ProjectBaseError)


def test_get_logger_returns_logger(monkeypatch):
    """get_logger must return a logging.Logger instance in dev mode."""
    monkeypatch.setenv("ENV", "development")
    import logging as _logging

    _logging.root.manager.loggerDict.pop("test_logger_dev", None)
    from src.logger import get_logger

    log = get_logger("test_logger_dev")
    assert isinstance(log, logging.Logger)


def test_get_logger_no_duplicate_handlers(monkeypatch):
    """Calling get_logger twice with same name must not duplicate handlers."""
    monkeypatch.setenv("ENV", "development")
    from src.logger import get_logger

    log1 = get_logger("test_logger_dedup")
    handler_count = len(log1.handlers)
    log2 = get_logger("test_logger_dedup")
    assert log1 is log2
    assert len(log2.handlers) == handler_count


def test_base_model_is_abstract():
    """BaseMLModel must be abstract and not instantiable directly."""
    from src.models.base import BaseMLModel

    with pytest.raises(TypeError):
        BaseMLModel()  # type: ignore[abstract]


def test_base_model_concrete_subclass():
    """A concrete subclass implementing all abstractmethods must instantiate."""
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from src.models.base import BaseMLModel

    class ConcreteModel(BaseMLModel):
        def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
            pass

        def predict(self, texts: list) -> list:
            return []

        def predict_proba(self, texts: list) -> np.ndarray:
            return np.array([])

        def save(self, path: Path) -> None:
            pass

        def load(self, path: Path) -> "ConcreteModel":
            return self

    model = ConcreteModel()
    assert isinstance(model, BaseMLModel)
