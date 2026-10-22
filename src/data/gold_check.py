"""Validate your hand-written gold set and check it does not leak into the training data.

    python -m src.data.gold_check
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from src.data.text import dataset_orthography


def check(gold_path: str | Path, train_path: str | Path | None = None) -> dict:
    gold = pd.read_csv(gold_path)
    missing = {"egyptian", "msa"} - set(gold.columns)
    if missing:
        raise ValueError(f"gold file is missing columns: {sorted(missing)}")
    gold["egyptian"], gold["msa"] = gold["egyptian"].astype("string"), gold["msa"].astype("string")
    empty = int((gold["egyptian"].isna() | gold["msa"].isna() | (gold["egyptian"].str.strip() == "")
                 | (gold["msa"].str.strip() == "")).sum())
    gold = gold.dropna(subset=["egyptian", "msa"])
    e_keys = gold["egyptian"].map(dataset_orthography)
    m_keys = gold["msa"].map(dataset_orthography)
    result = {
        "rows": len(gold), "empty_rows": empty,
        "duplicate_egyptian": int(e_keys.duplicated().sum()),
        "identical_pairs": int((e_keys == m_keys).sum()),
        "genres": gold["genre"].value_counts().to_dict() if "genre" in gold else {},
        "avg_words_egyptian": round(float(gold["egyptian"].str.split().str.len().mean()), 1) if len(gold) else 0.0,
        "train_overlap": None,
    }
    if train_path and Path(train_path).exists():
        train = [json.loads(line) for line in Path(train_path).read_text(encoding="utf-8").splitlines() if line.strip()]
        train_e, train_m = {r["egyptian"] for r in train}, {r["msa"] for r in train}
        result["train_overlap"] = int((e_keys.isin(train_e) | m_keys.isin(train_m)).sum())
    warnings = []
    if result["rows"] < 200:
        warnings.append(f"only {result['rows']} rows; aim for 200-300")
    if result["genres"] and max(result["genres"].values()) > 0.7 * result["rows"]:
        warnings.append("one genre is more than 70% of the set; diversify")
    if result["train_overlap"]:
        warnings.append(f"{result['train_overlap']} rows also appear in training data: replace them")
    if result["identical_pairs"] > 0.1 * max(result["rows"], 1):
        warnings.append("many rows have identical Egyptian and MSA text")
    result["warnings"] = warnings
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="data/gold/egy_msa_gold.csv", type=Path)
    parser.add_argument("--train", default="data/processed/egy_msa/train.jsonl", type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.gold, args.train), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
