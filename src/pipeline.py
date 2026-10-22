"""End-to-end dialect identification and conditional normalization."""
from pathlib import Path
from typing import Any, Callable
import json
import math
import joblib
from src.data.text import normalize_arabic
from src.normalizer.rules import normalize_to_msa


class DialectPipeline:
    def __init__(self, model: Any, *, egyptian_label: str = "Egyptian", threshold: float = 0.7,
                 normalizer: Callable[[str], str] = normalize_to_msa):
        self.model = model
        self.normalizer = normalizer
        self.egyptian_label = egyptian_label
        self.threshold = threshold

    @classmethod
    def from_path(cls, path: str | Path, **kwargs: Any) -> "DialectPipeline":
        """Load a saved model; use the validation-selected threshold.json next to it unless one is passed."""
        path = Path(path)
        threshold_file = path.with_name("threshold.json")
        if "threshold" not in kwargs and threshold_file.exists():
            kwargs["threshold"] = float(json.loads(threshold_file.read_text(encoding="utf-8"))["threshold"])
        return cls(joblib.load(path), **kwargs)

    def identify(self, text: str) -> dict[str, Any]:
        cleaned = normalize_arabic(text)
        label = str(self.model.predict([cleaned])[0])
        confidence = 1.0
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba([cleaned])[0]
            confidence = float(max(probabilities))
        elif hasattr(self.model, "decision_function"):
            scores = self.model.decision_function([cleaned])[0]
            if hasattr(scores, "__len__"):
                confidence = float(max(scores) / (sum(abs(float(s)) for s in scores) + 1e-9))
            else:
                # LinearSVC is not calibrated. This is a monotonic margin
                # heuristic; fit a calibrated classifier before using it as
                # a production threshold.
                confidence = 1.0 / (1.0 + math.exp(-min(50.0, max(-50.0, abs(float(scores))))))
        return {"dialect": label, "confidence": max(0.0, min(1.0, confidence))}

    def run(self, text: str) -> dict[str, Any]:
        result = self.identify(text)
        result["msa"] = (self.normalizer(text) if result["dialect"] == self.egyptian_label
                         and result["confidence"] >= self.threshold else None)
        return result

