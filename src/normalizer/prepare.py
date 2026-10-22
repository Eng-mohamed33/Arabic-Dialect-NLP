"""Prepare leakage-aware Egyptian-to-MSA parallel splits."""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data.load import load_parallel_text
from src.data.text import normalize_arabic


def _clean_pair(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["egyptian_raw"] = result["egyptian"].astype("string").str.strip()
    result["msa_raw"] = result["msa"].astype("string").str.strip()
    result["egyptian"] = result["egyptian_raw"].map(normalize_arabic)
    result["msa"] = result["msa_raw"].map(normalize_arabic)
    result = result[(result["egyptian"] != "") & (result["msa"] != "")]
    return result.drop_duplicates(subset=["egyptian", "msa"]).reset_index(drop=True)


def _load_gold(path: str | Path | None) -> tuple[set[str], set[str]]:
    if path is None or not Path(path).exists():
        return set(), set()
    gold = pd.read_csv(path)
    if not {"egyptian", "msa"} <= set(gold.columns) or gold.empty:
        return set(), set()
    return (set(gold["egyptian"].astype(str).map(normalize_arabic)),
            set(gold["msa"].astype(str).map(normalize_arabic)))


def prepare_splits(train_path: str | Path, valid_path: str | Path, test_path: str | Path,
                   output_dir: str | Path, gold_path: str | Path | None = None) -> dict:
    raw = {
        "train": load_parallel_text(train_path[0], train_path[1]) if isinstance(train_path, tuple) else None,
        "valid": load_parallel_text(valid_path[0], valid_path[1]) if isinstance(valid_path, tuple) else None,
        "test": load_parallel_text(test_path[0], test_path[1]) if isinstance(test_path, tuple) else None,
    }
    if any(value is None for value in raw.values()):
        raise TypeError("split paths must be (egyptian_path, msa_path) tuples")

    cleaned = {name: _clean_pair(frame) for name, frame in raw.items()}
    gold_sources, gold_targets = _load_gold(gold_path)
    test_sources = set(cleaned["test"]["egyptian"]) | gold_sources
    valid_sources = set(cleaned["valid"]["egyptian"])
    held_targets = set(cleaned["test"]["msa"]) | set(cleaned["valid"]["msa"]) | gold_targets
    valid_before = len(cleaned["valid"])
    cleaned["valid"] = cleaned["valid"][~cleaned["valid"]["egyptian"].isin(test_sources)].reset_index(drop=True)
    train_before = len(cleaned["train"])
    source_leak = cleaned["train"]["egyptian"].isin(test_sources | valid_sources)
    target_leak = cleaned["train"]["msa"].isin(held_targets)  # same MSA sentence in train and test lets a model memorize it
    msa_only = int((target_leak & ~source_leak).sum())
    cleaned["train"] = cleaned["train"][~(source_leak | target_leak)].reset_index(drop=True)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for name, frame in cleaned.items():
        frame[["egyptian_raw", "msa_raw", "egyptian", "msa"]].to_json(
            destination / f"{name}.jsonl", orient="records", lines=True, force_ascii=False
        )

    audit = {
        "raw_rows": {name: len(frame) for name, frame in raw.items()},
        "clean_rows": {name: len(frame) for name, frame in cleaned.items()},
        "train_removed_for_leakage": train_before - len(cleaned["train"]),
        "train_removed_msa_side_only": msa_only,
        "valid_removed_for_test_overlap": valid_before - len(cleaned["valid"]),
        "gold_rows": len(gold_sources),
        "source": "EGY_MSA_Translation",
    }
    (destination / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/egy_msa_translation"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/egy_msa"))
    parser.add_argument("--gold", type=Path, default=Path("data/gold/egy_msa_gold.csv"))
    args = parser.parse_args()
    raw = args.raw_dir
    audit = prepare_splits(
        (raw / "EGY-parallel_train_60955.txt", raw / "msa_parallel_train_52659.txt"),
        (raw / "EGY-parallel_valid_13358.txt", raw / "msa_parallel_valid_45049.txt"),
        (raw / "EGY-parallel_test_36402.txt", raw / "msa_parallel_test_44839.txt"),
        args.output_dir,
        args.gold,
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

