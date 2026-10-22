"""Collect every metrics file into one results page: reports/RESULTS.md

    python -m src.eval.summarize
"""
import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def collect_classifiers(reports: Path) -> list[dict]:
    sources = [("TF-IDF char n-grams + LinearSVC", reports / "dialect_baseline" / "metrics.json"),
               ("TF-IDF + LinearSVC (calibrated)", reports / "calibrated" / "metrics.json")]
    sources += [(folder.name, folder / "metrics.json") for folder in sorted((reports / "transformers").glob("*"))
                if folder.is_dir()]
    entries = []
    for name, path in sources:
        data = _load(path)
        if data:
            entries.append({"name": name, "macro_f1": data["macro_f1"], "accuracy": data["accuracy"],
                            "f1": {label: values["f1-score"] for label, values in data["per_class"].items()},
                            "test_rows": data.get("sizes", {}).get("test")})
    return sorted(entries, key=lambda e: -e["macro_f1"])


def render(reports: Path) -> str:
    lines = ["# Results", "", "## Dialect identification (same stratified split, seed 42)", ""]
    entries = collect_classifiers(reports)
    if entries:
        labels = sorted({label for e in entries for label in e["f1"]})
        lines += ["| Model | Macro-F1 | Accuracy | " + " | ".join(f"F1 {l}" for l in labels) + " |",
                  "|---|---|---|" + "---|" * len(labels)]
        for e in entries:
            lines.append(f"| {e['name']} | {e['macro_f1']:.4f} | {e['accuracy']:.4f} | "
                         + " | ".join(f"{e['f1'].get(l, float('nan')):.3f}" for l in labels) + " |")
    else:
        lines.append("_No classifier metrics found yet._")
    lines += ["", "## Egyptian → MSA (sacrebleu, same backend for every row)", "",
              "| Test set | System | BLEU | chrF |", "|---|---|---|---|"]
    found = False
    for title, path in (("held-out test", reports / "mt5_test.json"), ("hand-written gold", reports / "mt5_gold.json")):
        data = _load(path)
        if data:
            found = True
            lines += [f"| {title} ({data['rows']} rows) | Identity (copy input) | {data['identity']['bleu']:.2f} | {data['identity']['chrf']:.2f} |",
                      f"| {title} ({data['rows']} rows) | mT5-small fine-tuned | {data['model']['bleu']:.2f} | {data['model']['chrf']:.2f} |"]
    if not found:
        lines.append("| _none yet_ | | | |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", default="reports", type=Path)
    args = parser.parse_args()
    text = render(args.reports_dir)
    (args.reports_dir / "RESULTS.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
