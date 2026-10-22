from pathlib import Path

import pandas as pd
import pytest

from src.data.load import load_dataset


def test_loader_canonicalizes_labels_and_deduplicates(tmp_path: Path):
    path = tmp_path / "data.csv"
    pd.DataFrame({"text": ["أهلا", "أهلا", "مساء الخير"], "label": ["مصرى", "Egyptian", "MSA"]}).to_csv(path, index=False)
    result = load_dataset(path, labels=["Egyptian", "MSA"])
    assert len(result) == 2
    assert set(result["label"]) == {"Egyptian", "MSA"}
    assert "text_raw" in result


def test_loader_rejects_conflicting_normalized_duplicates(tmp_path: Path):
    path = tmp_path / "data.csv"
    pd.DataFrame({"text": ["أهلا", "اهلا"], "label": ["Egyptian", "MSA"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="conflicting labels"):
        load_dataset(path)

