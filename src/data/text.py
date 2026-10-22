"""Conservative Arabic text normalization."""
import re

_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MENTION = re.compile(r"[@#][\w\u0600-\u06FF]+")
_REPEATED = re.compile(r"(.)\1{2,}")


def normalize_arabic(text: str, *, remove_punctuation: bool = False) -> str:
    """Normalize common social-text noise while preserving word boundaries."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    value = _URL.sub(" ", text.strip())
    value = _MENTION.sub(" ", value)
    value = _DIACRITICS.sub("", value)
    value = value.translate(str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا"}))
    value = value.replace("ـ", "")
    if remove_punctuation:
        value = re.sub(r"[^\w\u0600-\u06FF\s]", " ", value)
    value = _REPEATED.sub(r"\1\1", value)
    return re.sub(r"\s+", " ", value).strip()



_DATASET_ORTHOGRAPHY = str.maketrans({"ة": "ه", "ى": "ي"})


def dataset_orthography(text: str) -> str:
    """normalize_arabic plus the ة->ه and ى->ي mapping already present in the EGY->MSA training data.

    The training files contain no ة or ى at all, so inputs and references must be mapped the same way
    before generation and scoring.
    """
    return normalize_arabic(text).translate(_DATASET_ORTHOGRAPHY)
