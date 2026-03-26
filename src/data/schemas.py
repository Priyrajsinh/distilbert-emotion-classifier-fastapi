"""Pydantic request/response schemas for the sentiment prediction API.

SentimentInput validates incoming text payloads.
SentimentOutput defines the structure of every prediction response.
"""

from pydantic import BaseModel, validator


class SentimentInput(BaseModel):
    """Validated input payload for the /predict endpoint.

    Attributes:
        text: The raw text string to classify. Must not be empty or
              whitespace-only.
    """

    text: str

    @validator("text")
    def not_empty(cls, value: str) -> str:
        """Reject blank or whitespace-only text inputs.

        Args:
            value: The raw text value provided by the caller.

        Returns:
            The original value unchanged if it passes validation.

        Raises:
            ValueError: If the stripped text is an empty string.
        """
        if value.strip() == "":
            raise ValueError("text must not be empty or whitespace-only")
        return value


class SentimentOutput(BaseModel):
    """Structured prediction response returned by the /predict endpoint.

    Attributes:
        label: Human-readable emotion label (e.g. 'joy', 'anger').
        confidence: Probability of the predicted class in [0.0, 1.0].
        probabilities: Mapping of every emotion label to its probability.
        trace_id: UUID string for request tracing and log correlation.
    """

    label: str
    confidence: float
    probabilities: dict[str, float]
    trace_id: str
