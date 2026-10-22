"""Build clean, leakage-checked datasets from the raw files.

    python -m src.data.prepare

Writes (all git-ignored):
    data/processed/dialect.csv                text,label,source,country
    data/processed/parallel_{train,valid,test}.csv   egyptian,msa,identical
and reports/data_audit.json + reports/data_audit.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.load import load_parallel_text
from src.data.text import normalize_arabic

IADD_LABELS = {"EGY": "Egyptian", "GLF": "Gulf", "LEV": "Levantine", "MGH": "Maghrebi", "IRQ": "Iraqi"}
PARALLEL_FILES = {
    "train": ("EGY-parallel_train_60955.txt", "msa_parallel_train_52659.txt"),
    "valid": ("EGY-parallel_valid_13358.txt", "msa_parallel_valid_45049.txt"),
    "test": ("EGY-parallel_test_36402.txt", "msa_parallel_test_44839.txt"),
}


def prepare_dialect(iadd_path: str | Path, *, min_class_count: int = 500, max_words: int = 200,
                    extra_msa: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean IADD: drop `general`, empty/very long rows, conflicting and duplicate texts,
    and classes too small to split and evaluate reliably."""
    records = json.loads(Path(iadd_path).read_text(encoding="utf-8"))
    frame = pd.DataFrame(records).rename(columns={"Sentence": "text", "Region": "label",
                                                  "DataSource": "source", "Country": "country"})
    info: dict[str, Any] = {"raw_rows": len(frame)}
    info["general_removed"] = int((frame["label"] == "general").sum())
    frame = frame[frame["label"] != "general"].copy()
    frame["label"] = frame["label"].map(IADD_LABELS)
    frame["text"] = frame["text"].astype("string").str.strip()
    frame = frame[["text", "label", "source", "country"]]
    if extra_msa is not None:
        frame = pd.concat([frame, extra_msa], ignore_index=True)

    frame["key"] = frame["text"].map(normalize_arabic)
    before = len(frame)
    frame = frame[frame["key"] != ""]
    info["empty_removed"] = before - len(frame)

    before = len(frame)
    frame = frame[frame["key"].str.split().str.len() <= max_words]
    info["too_long_removed"] = before - len(frame)

    conflicts = frame.groupby("key")["label"].nunique()
    conflicting = set(conflicts[conflicts > 1].index)
    info["conflicting_texts_dropped"] = len(conflicting)
    frame = frame[~frame["key"].isin(conflicting)]

    before = len(frame)
    frame = frame.drop_duplicates(subset="key")
    info["duplicates_removed"] = before - len(frame)

    counts = frame["label"].value_counts()
    small = sorted(counts[counts < min_class_count].index)
    info["classes_dropped_too_small"] = {label: int(counts[label]) for label in small}
    frame = frame[~frame["label"].isin(small)]

    frame = frame.drop(columns="key").reset_index(drop=True)
    info["final_rows"] = len(frame)
    info["label_counts"] = {k: int(v) for k, v in frame["label"].value_counts().items()}
    info["label_by_source"] = {
        label: {k: int(v) for k, v in group["source"].value_counts().items()}
        for label, group in frame.groupby("label")
    }
    return frame, info


def _clean_parallel(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["e_key"] = frame["egyptian"].map(normalize_arabic)
    frame["m_key"] = frame["msa"].map(normalize_arabic)
    frame = frame[(frame["e_key"] != "") & (frame["m_key"] != "")]
    return frame.drop_duplicates(subset=["e_key", "m_key"]).reset_index(drop=True)


def prepare_parallel(directory: str | Path, *, gold_path: str | Path | None = None
                     ) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Load train/valid/test, dedupe, and remove every train/valid pair whose Egyptian OR MSA side
    also appears in a held-out set (valid, test, or your gold set)."""
    base = Path(directory)
    raw = {name: load_parallel_text(base / e, base / m) for name, (e, m) in PARALLEL_FILES.items()}
    info: dict[str, Any] = {"raw_rows": {k: len(v) for k, v in raw.items()}}
    clean = {k: _clean_parallel(v) for k, v in raw.items()}
    info["after_dedupe"] = {k: len(v) for k, v in clean.items()}

    gold_e: set[str] = set()
    gold_m: set[str] = set()
    if gold_path and Path(gold_path).exists():
        gold = pd.read_csv(gold_path)
        if {"egyptian", "msa"} <= set(gold.columns) and len(gold):
            gold_e = set(gold["egyptian"].astype(str).map(normalize_arabic))
            gold_m = set(gold["msa"].astype(str).map(normalize_arabic))
    info["gold_rows"] = len(gold_e)

    # the test set is the reference: nothing in it may appear anywhere else
    test_e, test_m = set(clean["test"]["e_key"]) | gold_e, set(clean["test"]["m_key"]) | gold_m
    before = len(clean["valid"])
    clean["valid"] = clean["valid"][~(clean["valid"]["e_key"].isin(test_e) | clean["valid"]["m_key"].isin(test_m))]
    info["valid_removed_overlap_test"] = before - len(clean["valid"])

    held_e = test_e | set(clean["valid"]["e_key"])
    held_m = test_m | set(clean["valid"]["m_key"])
    before = len(clean["train"])
    clean["train"] = clean["train"][~(clean["train"]["e_key"].isin(held_e) | clean["train"]["m_key"].isin(held_m))]
    info["train_removed_leakage"] = before - len(clean["train"])

    out = {}
    for name, frame in clean.items():
        frame = frame.reset_index(drop=True)
        frame["identical"] = frame["e_key"] == frame["m_key"]
        out[name] = frame[["egyptian", "msa", "identical"]]
    info["final_rows"] = {k: len(v) for k, v in out.items()}
    info["identical_share"] = {k: round(float(v["identical"].mean()), 4) for k, v in out.items()}
    return out, info


def write_audit(audit: dict[str, Any], reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "data_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    d, p = audit["dialect"], audit["parallel"]
    lines = ["# Data audit", "", "## Dialect identification (IADD)", ""]
    lines += [f"- raw rows: {d['raw_rows']}", f"- `general` removed: {d['general_removed']}",
              f"- empty removed: {d['empty_removed']}", f"- too long removed: {d['too_long_removed']}",
              f"- conflicting texts dropped: {d['conflicting_texts_dropped']}",
              f"- duplicates removed: {d['duplicates_removed']}",
              f"- classes dropped (too small): {d['classes_dropped_too_small']}",
              f"- final rows: {d['final_rows']}", "", "| label | rows | sources |", "|---|---|---|"]
    for label, count in d["label_counts"].items():
        lines.append(f"| {label} | {count} | {d['label_by_source'][label]} |")
    lines += ["", "> If a class comes almost entirely from one source, a model can learn the source's style",
              "> instead of the dialect. Report cross-source results and review errors by source.", "",
              "## Egyptian -> MSA (EGY_MSA_Translation)", ""]
    lines += [f"- raw rows: {p['raw_rows']}", f"- after dedupe: {p['after_dedupe']}",
              f"- gold rows checked for leakage: {p['gold_rows']}",
              f"- valid removed (overlap with test/gold): {p['valid_removed_overlap_test']}",
              f"- train removed (leakage): {p['train_removed_leakage']}", f"- final rows: {p['final_rows']}",
              f"- identical EGY==MSA share: {p['identical_share']}"]
    (reports_dir / "data_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--iadd", default="data/raw/IADD.json", type=Path)
    parser.add_argument("--parallel-dir", default="data/raw/egy_msa_translation", type=Path)
    parser.add_argument("--gold", default="data/gold/egy_msa_gold.csv", type=Path)
    parser.add_argument("--out-dir", default="data/processed", type=Path)
    parser.add_argument("--reports-dir", default="reports", type=Path)
    parser.add_argument("--min-class-count", default=500, type=int)
    parser.add_argument("--max-words", default=200, type=int)
    parser.add_argument("--msa-from-parallel", default=0, type=int, metavar="N",
                        help="add N MSA sentences taken from the parallel corpus as an MSA class (off by default:\n"
                             "its domain differs from IADD, so a model can separate it by style alone)")
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    parallel, parallel_info = prepare_parallel(args.parallel_dir, gold_path=args.gold)
    extra = None
    if args.msa_from_parallel > 0:
        sample = parallel["train"]["msa"].sample(n=args.msa_from_parallel, random_state=args.seed)
        extra = pd.DataFrame({"text": sample.values, "label": "MSA",
                              "source": "EGY_MSA_Translation(MSA side)", "country": "NA"})
    dialect, dialect_info = prepare_dialect(args.iadd, min_class_count=args.min_class_count,
                                            max_words=args.max_words, extra_msa=extra)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dialect.to_csv(args.out_dir / "dialect.csv", index=False)
    for name, frame in parallel.items():
        frame.to_csv(args.out_dir / f"parallel_{name}.csv", index=False)
    write_audit({"dialect": dialect_info, "parallel": parallel_info}, args.reports_dir)
    print(json.dumps({"dialect": dialect_info["label_counts"], "dropped": dialect_info["classes_dropped_too_small"],
                      "parallel": parallel_info["final_rows"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
