import json
from pathlib import Path

import pandas as pd

from src.dialect_id.train_transformer import MODELS, cap_per_class, present_labels
from src.eval.summarize import render


def _metrics(path: Path, macro: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"macro_f1": macro, "accuracy": macro + 0.1, "sizes": {"test": 10},
                                "per_class": {"Egyptian": {"f1-score": macro}, "Gulf": {"f1-score": macro}}}), encoding="utf-8")


def test_summarize_sorts_models_and_includes_translation(tmp_path: Path):
    _metrics(tmp_path / "dialect_baseline" / "metrics.json", 0.8)
    _metrics(tmp_path / "transformers" / "marbert" / "metrics.json", 0.9)
    (tmp_path / "mt5_test.json").write_text(json.dumps({"rows": 5, "identity": {"bleu": 1.0, "chrf": 2.0},
                                                        "model": {"bleu": 3.0, "chrf": 4.0}}), encoding="utf-8")
    text = render(tmp_path)
    assert text.index("marbert") < text.index("LinearSVC") and "mT5-small" in text and "3.00" in text


def test_transformer_helpers():
    assert {"arabert", "marbert", "camelbert"} <= set(MODELS)
    assert present_labels(["Gulf", "Egyptian", "Gulf"]) == ["Egyptian", "Gulf"]
    frame = pd.DataFrame({"text": [str(i) for i in range(30)], "label": ["Egyptian"] * 20 + ["Gulf"] * 10})
    capped = cap_per_class(frame, 5, seed=1)
    assert capped["label"].value_counts().to_dict() == {"Egyptian": 5, "Gulf": 5}
