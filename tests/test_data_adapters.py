import json
from pathlib import Path

import pytest

from src.data.load import load_iadd, load_parallel_text


def test_iadd_adapter_maps_regions_and_drops_general(tmp_path: Path):
    path = tmp_path / "IADD.json"
    path.write_text(json.dumps([
        {"Sentence": "ازيك", "Region": "EGY"},
        {"Sentence": "شلونك", "Region": "GLF"},
        {"Sentence": "نص عام", "Region": "general"},
    ], ensure_ascii=False), encoding="utf-8")
    result = load_iadd(path)
    assert list(result["label"]) == ["Egyptian", "Gulf"]


def test_parallel_loader_rejects_misaligned_files(tmp_path: Path):
    eg = tmp_path / "eg.txt"
    msa = tmp_path / "msa.txt"
    eg.write_text("واحد\nاتنين\n", encoding="utf-8")
    msa.write_text("واحد\n", encoding="utf-8")
    with pytest.raises(ValueError, match="different lengths"):
        load_parallel_text(eg, msa)

