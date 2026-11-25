"""Public entry points for prediction and scoring."""
from .prediction import predict
from .scoring.pipeline import score

__all__ = ["predict", "score"]
