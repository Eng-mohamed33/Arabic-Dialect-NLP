"""Validation-only threshold selection helpers."""

from typing import Iterable

import numpy as np
from sklearn.metrics import f1_score


def select_threshold(y_true: Iterable[str], positive_scores: Iterable[float],
                     *, positive_label: str = "Egyptian",
                     candidates: Iterable[float] | None = None) -> dict[str, float]:
    """Select the threshold maximizing positive-class F1 on validation data."""
    actual = np.asarray(list(y_true))
    scores = np.asarray(list(positive_scores), dtype=float)
    if len(actual) != len(scores) or len(actual) == 0:
        raise ValueError("y_true and positive_scores must be non-empty and have equal length")
    choices = np.asarray(list(candidates) if candidates is not None else np.arange(0.05, 1.0, 0.01))
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in choices:
        predicted = np.where(scores >= threshold, positive_label, "not_positive")
        f1 = f1_score(actual == positive_label, predicted == positive_label, zero_division=0)
        if f1 > best_f1:
            best_threshold, best_f1 = float(threshold), float(f1)
    return {"threshold": best_threshold, "f1": best_f1}

