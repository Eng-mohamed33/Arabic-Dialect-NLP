import json
from pathlib import Path

import pandas as pd

from src.data.gold_check import check
from src.data.text import dataset_orthography
from src.normalizer.evaluate import build_report, load_rows


def test_dataset_orthography_maps_ta_marbuta_and_alef_maqsura():
    assert dataset_orthography("مدرسة إلى") == "مدرسه الي"


def test_load_rows_maps_csv_but_not_jsonl(tmp_path: Path):
    csv = tmp_path / "gold.csv"
    pd.DataFrame({"egyptian": ["رايح المدرسة"], "msa": ["ذاهب إلى المدرسة"]}).to_csv(csv, index=False)
    assert load_rows(csv)[0]["msa"] == "ذاهب الي المدرسه"
    jsonl = tmp_path / "test.jsonl"
    jsonl.write_text(json.dumps({"egyptian": "مدرسة", "msa": "مدرسة"}, ensure_ascii=False) + "\n", encoding="utf-8")
    assert load_rows(jsonl)[0]["msa"] == "مدرسة"


def test_build_report_has_per_genre_scores():
    sources = [f"جملة مصريه {i}" for i in range(10)]
    references = [f"جمله فصحى {i}" for i in range(10)]
    genres = ["daily"] * 5 + ["formal"] * 5
    report = build_report(sources, references, references, genres)
    assert set(report["by_genre"]) == {"daily", "formal"}
    assert report["by_genre"]["daily"]["model_chrf"] > report["by_genre"]["daily"]["identity_chrf"]


def test_gold_check_flags_small_sets_and_train_overlap(tmp_path: Path):
    gold = tmp_path / "gold.csv"
    pd.DataFrame({"egyptian": ["ازيك", "رايح فين"], "msa": ["كيف حالك", "الي اين"], "genre": ["a", "a"]}).to_csv(gold, index=False)
    train = tmp_path / "train.jsonl"
    train.write_text(json.dumps({"egyptian": "ازيك", "msa": "x"}, ensure_ascii=False) + "\n", encoding="utf-8")
    result = check(gold, train)
    assert result["rows"] == 2 and result["train_overlap"] == 1
    assert any("200" in w for w in result["warnings"]) and any("training" in w for w in result["warnings"])
