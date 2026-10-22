"""Classification metrics and report serialization."""

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def classification_metrics(y_true: Any, y_pred: Any, labels: list[str]) -> dict[str, Any]:
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return {
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "per_class": {
            label: {metric: report[label][metric] for metric in ("precision", "recall", "f1-score", "support")}
            for label in labels
        },
        "labels": labels,
        "confusion_matrix": matrix,
    }


def save_report(report: dict[str, Any], output_dir: str | Path) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    matrix = pd.DataFrame(report["confusion_matrix"], index=report["labels"], columns=report["labels"])
    matrix.to_csv(destination / "confusion_matrix.csv", encoding="utf-8-sig")

