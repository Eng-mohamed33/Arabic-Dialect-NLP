from pathlib import Path

from src.normalizer.prepare import prepare_splits


def _write(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_prepare_splits_removes_cross_split_source_leakage(tmp_path: Path):
    train_eg, train_msa = tmp_path / "train.eg", tmp_path / "train.msa"
    valid_eg, valid_msa = tmp_path / "valid.eg", tmp_path / "valid.msa"
    test_eg, test_msa = tmp_path / "test.eg", tmp_path / "test.msa"
    _write(train_eg, ["ازيك", "عامل ايه", "جديد"])
    _write(train_msa, ["كيف حالك", "كيف حالك", "جديد"])
    _write(valid_eg, ["عامل ايه"])
    _write(valid_msa, ["كيف حالك"])
    _write(test_eg, ["ازيك"])
    _write(test_msa, ["كيف حالك"])
    audit = prepare_splits((train_eg, train_msa), (valid_eg, valid_msa), (test_eg, test_msa), tmp_path / "out")
    assert audit["clean_rows"] == {"train": 1, "valid": 1, "test": 1}
    assert audit["train_removed_for_leakage"] == 2



def test_prepare_splits_removes_msa_side_leakage_and_gold(tmp_path: Path):
    import pandas as pd
    files = {"train": (["جملة ا", "جملة ب", "جملة ج"], ["هدف مشترك", "هدف ب", "هدف ج"]),
             "valid": (["جملة د"], ["هدف د"]), "test": (["جملة هـ"], ["هدف مشترك"])}
    paths = {}
    for name, (eg, msa) in files.items():
        paths[name] = (tmp_path / f"{name}.eg", tmp_path / f"{name}.msa")
        _write(paths[name][0], eg)
        _write(paths[name][1], msa)
    gold = tmp_path / "gold.csv"
    pd.DataFrame({"egyptian": ["جملة ج"], "msa": ["مرجع"]}).to_csv(gold, index=False)
    audit = prepare_splits(paths["train"], paths["valid"], paths["test"], tmp_path / "out", gold)
    assert audit["clean_rows"]["train"] == 1  # shared-target row and gold row removed
    assert audit["train_removed_msa_side_only"] == 1
