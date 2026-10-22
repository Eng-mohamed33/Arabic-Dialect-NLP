"""Transparent placeholder; replace with identity/LLM/seq2seq experiments."""
from src.data.text import normalize_arabic


def normalize_to_msa(text: str) -> str:
    return normalize_arabic(text)

