"""Probability-calibrated dialect classifier."""

from sklearn.calibration import CalibratedClassifierCV

from src.dialect_id.baseline import build_model


def build_calibrated_model(*, cv: int = 3) -> CalibratedClassifierCV:
    """Wrap the LinearSVC baseline so its scores can be treated as probabilities."""
    return CalibratedClassifierCV(estimator=build_model(), method="sigmoid", cv=cv)

