"""Leakage-aware dataset splitting helpers."""
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_split(frame: pd.DataFrame, *, label_col: str = "label", seed: int = 42,
                     train_size: float = 0.8, validation_size: float = 0.1
                     ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return train/validation/test with stable stratification."""
    if not 0 < train_size < 1 or not 0 < validation_size < 1 or train_size + validation_size >= 1:
        raise ValueError("train_size and validation_size must leave a positive test split")
    if label_col not in frame or frame[label_col].isna().any():
        raise ValueError(f"missing or null label column: {label_col}")
    train, rest = train_test_split(frame, test_size=1 - train_size, random_state=seed,
                                   stratify=frame[label_col])
    relative_validation = validation_size / (1 - train_size)
    validation, test = train_test_split(rest, test_size=1 - relative_validation,
                                        random_state=seed, stratify=rest[label_col])
    return train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True)

