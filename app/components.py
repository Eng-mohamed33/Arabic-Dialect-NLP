"""Shared model loading for the FastAPI service and the Streamlit demo."""
import os
from pathlib import Path
from typing import Any, Callable

from src.normalizer.rules import normalize_to_msa
from src.pipeline import DialectPipeline


def resolve_dialect_path() -> Path:
    """DIALECT_MODEL_PATH, else the calibrated model if it exists, else the plain baseline."""
    if os.getenv("DIALECT_MODEL_PATH"):
        return Path(os.environ["DIALECT_MODEL_PATH"])
    for candidate in ("reports/calibrated/dialect_baseline.joblib", "reports/dialect_baseline/dialect_baseline.joblib"):
        if Path(candidate).exists():
            return Path(candidate)
    return Path("reports/dialect_baseline/dialect_baseline.joblib")


def resolve_normalizer_dir() -> Path:
    return Path(os.getenv("NORMALIZER_MODEL_DIR", "models/egy_msa_mt5"))


def download_from_hub(repo_id: str, target: str | Path = "hub_models") -> tuple[Path, Path]:
    """Fetch models from a Hugging Face model repo laid out as egy_msa_mt5/ and dialect/ (used on Spaces)."""
    from huggingface_hub import snapshot_download
    root = Path(snapshot_download(repo_id=repo_id, local_dir=str(target)))
    return root / "dialect" / "dialect_baseline.joblib", root / "egy_msa_mt5"


def load_normalizer(model_dir: Path) -> tuple[Callable[[str], str], str]:
    """Use the fine-tuned mT5 if its folder is present, otherwise the transparent rules placeholder."""
    if (model_dir / "config.json").exists():
        from src.normalizer.seq2seq import Seq2SeqNormalizer
        return Seq2SeqNormalizer(model_dir), "mt5"
    return normalize_to_msa, "rules"


def load_components(dialect_path: Path | None = None, normalizer_dir: Path | None = None
                    ) -> tuple[DialectPipeline | None, Callable[[str], str], str]:
    dialect_path, normalizer_dir = dialect_path or resolve_dialect_path(), normalizer_dir or resolve_normalizer_dir()
    if os.getenv("HF_MODEL_REPO") and not (normalizer_dir / "config.json").exists():
        dialect_path, normalizer_dir = download_from_hub(os.environ["HF_MODEL_REPO"])
    normalizer, backend = load_normalizer(normalizer_dir)
    pipeline: Any = DialectPipeline.from_path(dialect_path, normalizer=normalizer) if dialect_path.exists() else None
    return pipeline, normalizer, backend
