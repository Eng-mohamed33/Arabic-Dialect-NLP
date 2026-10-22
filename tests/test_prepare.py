import json
from pathlib import Path

import pandas as pd

from src.data.prepare import prepare_dialect, prepare_parallel


def _iadd(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "IADD.json"
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return path


def test_prepare_dialect_cleans_and_drops_small_classes(tmp_path: Path):
    rows = [{"Sentence": f"جملة مصرية {i}", "Region": "EGY", "DataSource": "AOC", "Country": "Egypt"} for i in range(5)]
    rows += [{"Sentence": f"شلونك {i}", "Region": "GLF", "DataSource": "AOC", "Country": "NA"} for i in range(5)]
    rows += [{"Sentence": "نص عام", "Region": "general", "DataSource": "AOC", "Country": "NA"},
             {"Sentence": "جملة مصرية 0", "Region": "EGY", "DataSource": "DART", "Country": "Egypt"},  # duplicate
             {"Sentence": "غامض", "Region": "EGY", "DataSource": "AOC", "Country": "NA"},
             {"Sentence": "غامض", "Region": "GLF", "DataSource": "AOC", "Country": "NA"},  # conflict
             {"Sentence": "عراقي", "Region": "IRQ", "DataSource": "DART", "Country": "Iraq"}]
    frame, info = prepare_dialect(_iadd(tmp_path, rows), min_class_count=5)
    assert info["general_removed"] == 1 and info["duplicates_removed"] == 1
    assert info["conflicting_texts_dropped"] == 1
    assert "Iraqi" in info["classes_dropped_too_small"]
    assert set(frame["label"]) == {"Egyptian", "Gulf"} and len(frame) == 10


def test_prepare_parallel_removes_train_test_leakage(tmp_path: Path):
    def write(name: str, lines: list[str]) -> None:
        (tmp_path / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    write("EGY-parallel_train_60955.txt", ["مصري ١", "مصري ٢", "مصري ٢", "مصري تست"])
    write("msa_parallel_train_52659.txt", ["فصحى ١", "فصحى ٢", "فصحى ٢", "فصحى مختلفة"])
    write("EGY-parallel_valid_13358.txt", ["مصري فاليد"])
    write("msa_parallel_valid_45049.txt", ["فصحى فاليد"])
    write("EGY-parallel_test_36402.txt", ["مصري تست"])
    write("msa_parallel_test_44839.txt", ["فصحى تست"])
    out, info = prepare_parallel(tmp_path)
    assert len(out["train"]) == 2  # one duplicate pair and one leaked pair removed
    assert info["train_removed_leakage"] == 1
    assert not set(out["train"]["egyptian"]) & set(out["test"]["egyptian"])


def test_prepare_parallel_uses_gold_set(tmp_path: Path):
    for name, line in [("EGY-parallel_train_60955.txt", "مصري ١"), ("msa_parallel_train_52659.txt", "فصحى ١"),
                       ("EGY-parallel_valid_13358.txt", "مصري ٢"), ("msa_parallel_valid_45049.txt", "فصحى ٢"),
                       ("EGY-parallel_test_36402.txt", "مصري ٣"), ("msa_parallel_test_44839.txt", "فصحى ٣")]:
        (tmp_path / name).write_text(line + "\n", encoding="utf-8")
    gold = tmp_path / "gold.csv"
    pd.DataFrame({"egyptian": ["مصري ١"], "msa": ["غير مهم"]}).to_csv(gold, index=False)
    out, info = prepare_parallel(tmp_path, gold_path=gold)
    assert len(out["train"]) == 0 and info["gold_rows"] == 1
