"""Train and evaluate the baseline from a local CSV/JSONL dataset."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from src.data.load import load_dataset
from src.data.split import stratified_split
from src.dialect_id.baseline import train
from src.dialect_id.calibrated import build_calibrated_model
from src.eval.metrics import classification_metrics, save_report
from src.eval.threshold import select_threshold


DEFAULT_LABELS = ["Egyptian", "Gulf", "Levantine", "Maghrebi", "MSA"]
ALLOWED_LABELS = DEFAULT_LABELS + ["Iraqi"]


def run(input_path: str | Path, output_dir: str | Path, seed: int = 42,
        calibrated: bool = False, calibration_cv: int = 3) -> dict:
    frame = load_dataset(input_path, labels=ALLOWED_LABELS)
    # score only classes present in the data: an absent class (e.g. no MSA) would add a fake 0-F1 to macro-F1
    labels = [label for label in ALLOWED_LABELS if label in set(frame["label"])]
    train_frame, validation_frame, test_frame = stratified_split(frame, seed=seed)
    if calibrated:
        model = build_calibrated_model(cv=calibration_cv)
        model.fit(train_frame["text"], train_frame["label"])
    else:
        model = train(train_frame)
    predictions = model.predict(test_frame["text"])
    report = classification_metrics(test_frame["label"], predictions, labels)
    report["seed"] = seed
    report["calibrated"] = calibrated
    report["sizes"] = {"all": len(frame), "train": len(train_frame), "validation": len(validation_frame), "test": len(test_frame)}
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination / "dialect_baseline.joblib")
    if calibrated:
        # tune the threshold for the rule the pipeline really applies: predicted Egyptian AND confidence >= threshold
        probabilities = model.predict_proba(validation_frame["text"])
        egyptian_index = list(model.classes_).index("Egyptian")
        predicted_egyptian = probabilities.argmax(axis=1) == egyptian_index
        validation_scores = np.where(predicted_egyptian, probabilities.max(axis=1), 0.0)
        calibration = select_threshold(validation_frame["label"], validation_scores)
        calibration["rule"] = "predicted Egyptian and confidence >= threshold"
        report["calibration"] = calibration
        (destination / "threshold.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    save_report(report, destination)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", default="reports/baseline", type=Path)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--calibrated", action="store_true", help="fit probability calibration and select Egyptian threshold")
    parser.add_argument("--calibration-cv", default=3, type=int)
    args = parser.parse_args()
    report = run(args.input, args.output_dir, args.seed, args.calibrated, args.calibration_cv)
    print(f"macro_f1={report['macro_f1']:.4f} accuracy={report['accuracy']:.4f}")


if __name__ == "__main__":
    main()

