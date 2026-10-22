"""Load and validate local dialect datasets without redistributing them."""

from pathlib import Path
from typing import Iterable
import json

import pandas as pd

from src.data.text import normalize_arabic

LABEL_ALIASES = {
    "egyptian": "Egyptian",
    "مصرى": "Egyptian",
    "مصري": "Egyptian",
    "gulf": "Gulf",
    "khaleeji": "Gulf",
    "خليجي": "Gulf",
    "levantine": "Levantine",
    "shami": "Levantine",
    "شامي": "Levantine",
    "maghrebi": "Maghrebi",
    "مغاربي": "Maghrebi",
    "msa": "MSA",
    "fusha": "MSA",
    "فصحى": "MSA",
}


def _read(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    raise ValueError("supported dataset formats are .csv, .jsonl, and .ndjson")


def canonical_label(label: object) -> str:
    value = str(label).strip()
    return LABEL_ALIASES.get(value.casefold(), value)


def load_dataset(path: str | Path, *, text_col: str = "text", label_col: str = "label",
                 labels: Iterable[str] | None = None) -> pd.DataFrame:
    """Load a dataset, canonicalize labels, and reject conflicting duplicates.

    The returned frame retains the original text in ``text_raw`` and stores the
    normalized model input in ``text``.
    """
    source = Path(path)
    frame = _read(source)
    missing = {text_col, label_col} - set(frame.columns)
    if missing:
        raise ValueError(f"dataset is missing columns: {sorted(missing)}")

    result = frame[[text_col, label_col]].rename(columns={text_col: "text_raw", label_col: "label"}).copy()
    result["text_raw"] = result["text_raw"].astype("string").str.strip()
    result["label"] = result["label"].map(canonical_label)
    result["text"] = result["text_raw"].map(normalize_arabic)
    result = result[(result["text_raw"].notna()) & (result["text"] != "") & result["label"].ne("")]

    if labels is not None:
        allowed = set(labels)
        unknown = sorted(set(result["label"]) - allowed)
        if unknown:
            raise ValueError(f"unexpected labels: {unknown}; expected one of {sorted(allowed)}")

    conflicts = result.groupby("text")["label"].nunique()
    if (conflicts > 1).any():
        examples = conflicts[conflicts > 1].index[:5].tolist()
        raise ValueError(f"normalized duplicate texts have conflicting labels, examples: {examples}")

    return result.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)


def load_iadd(path: str | Path) -> pd.DataFrame:
    """Adapt the public IADD JSON records to the project schema.

    ``general`` records are excluded because they are not one of the five
    target classes. Iraq remains a separate source label and should be mapped
    deliberately before a five-class experiment.
    """
    source = Path(path)
    records = json.loads(source.read_text(encoding="utf-8"))
    frame = pd.DataFrame(records)
    required = {"Sentence", "Region"}
    if not required.issubset(frame.columns):
        raise ValueError(f"IADD file is missing columns: {sorted(required - set(frame.columns))}")
    frame = frame.rename(columns={"Sentence": "text", "Region": "label"})
    frame = frame[frame["label"].ne("general")]
    frame["label"] = frame["label"].map({"EGY": "Egyptian", "GLF": "Gulf", "LEV": "Levantine", "MGH": "Maghrebi", "IRQ": "Iraqi"})
    return load_dataset_from_frame(frame, conflict_policy="drop")


def load_dataset_from_frame(frame: pd.DataFrame, *, text_col: str = "text", label_col: str = "label",
                            conflict_policy: str = "error") -> pd.DataFrame:
    """Run the same validation pipeline on an already-loaded frame."""
    missing = {text_col, label_col} - set(frame.columns)
    if missing:
        raise ValueError(f"dataset is missing columns: {sorted(missing)}")
    result = frame[[text_col, label_col]].rename(columns={text_col: "text_raw", label_col: "label"}).copy()
    result["text_raw"] = result["text_raw"].astype("string").str.strip()
    result["label"] = result["label"].map(canonical_label)
    result["text"] = result["text_raw"].map(normalize_arabic)
    result = result[(result["text_raw"].notna()) & (result["text"] != "") & result["label"].ne("")]
    conflicts = result.groupby("text")["label"].nunique()
    if (conflicts > 1).any():
        if conflict_policy == "drop":
            conflicting_texts = set(conflicts[conflicts > 1].index)
            result = result[~result["text"].isin(conflicting_texts)]
        else:
            raise ValueError("normalized duplicate texts have conflicting labels")
    elif conflict_policy not in {"error", "drop"}:
        raise ValueError("conflict_policy must be 'error' or 'drop'")
    return result.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)


def load_parallel_text(egyptian_path: str | Path, msa_path: str | Path) -> pd.DataFrame:
    """Load aligned EGY/MSA line files and reject line-count mismatches."""
    egyptian = Path(egyptian_path).read_text(encoding="utf-8").splitlines()
    msa = Path(msa_path).read_text(encoding="utf-8").splitlines()
    if len(egyptian) != len(msa):
        raise ValueError(f"parallel files have different lengths: {len(egyptian)} vs {len(msa)}")
    return pd.DataFrame({"egyptian": egyptian, "msa": msa})

