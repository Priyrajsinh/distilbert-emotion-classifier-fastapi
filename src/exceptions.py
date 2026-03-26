"""Custom exception hierarchy for the B1 project.

All project-specific exceptions inherit from ProjectBaseError so that
callers can catch any project error with a single except clause.
"""


class ProjectBaseError(Exception):
    """Base class for all B1 project exceptions.

    Inherit from this class for every project-specific error so that
    callers can catch the full hierarchy with a single except clause.
    """


class DataLoadError(ProjectBaseError):
    """Raised when raw or processed data cannot be loaded or parsed.

    Examples: missing CSV file, corrupt parquet, schema mismatch on disk.
    """


class ModelNotFoundError(ProjectBaseError):
    """Raised when a saved model artefact cannot be located on disk.

    Typically thrown by SentimentClassifier.load() when the expected
    model directory does not exist.
    """


class PredictionError(ProjectBaseError):
    """Raised when inference fails for a given input.

    This maps to HTTP 422 in the FastAPI layer via the registered
    exception handler in src/api/app.py.
    """


class ConfigError(ProjectBaseError):
    """Raised when config/config.yaml is missing, malformed, or incomplete.

    Thrown early at startup so misconfiguration is caught before any
    training or serving begins.
    """
