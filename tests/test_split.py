import pandas as pd
from src.data.split import stratified_split


def test_stratified_split_preserves_labels():
    frame = pd.DataFrame({"text": [f"x{i}" for i in range(20)], "label": ["a", "b"] * 10})
    train, validation, test = stratified_split(frame)
    assert len(train) + len(validation) + len(test) == len(frame)
    assert set(train.label) == set(validation.label) == set(test.label)

