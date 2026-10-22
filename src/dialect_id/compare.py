"""Compare the two lightweight baselines on an identical split."""

import argparse
import json
from pathlib import Path

from src.data.load import load_dataset
from src.data.split import stratified_split
from src.dialect_id.baseline import train
from src.eval.metrics import classification_metrics

LABELS = ["Egyptian", "Gulf", "Levantine", "Maghrebi", "MSA"]


def compare(input_path: str | Path, output_dir: str | Path, seed: int = 42) -> dict:
    frame = load_dataset(input_path, labels=LABELS)
    train_frame, _, test_frame = stratified_split(frame, seed=seed)
    results = {"seed": seed, "sizes": {"all": len(frame), "train": len(train_frame), "test": len(test_frame)}}
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    for model_type in ("linearsvc", "naive_bayes"):
        model = train(train_frame, model_type=model_type)
        predictions = model.predict(test_frame["text"])
        results[model_type] = classification_metrics(test_frame["label"], predictions, LABELS)

    (destination / "comparison.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", default="reports/comparison", type=Path)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    results = compare(args.input, args.output_dir, args.seed)
    for model_type in ("linearsvc", "naive_bayes"):
        print(f"{model_type}: macro_f1={results[model_type]['macro_f1']:.4f}")


if __name__ == "__main__":
    main()

