"""Dependency-light Identity baseline metrics for Egyptian -> MSA."""

import argparse
import json
import math
from collections import Counter
from pathlib import Path


def _ngram_counts(text: str | list[str], order: int) -> Counter:
    sequence = list(text) if isinstance(text, str) else text
    return Counter(
        sequence[i] if order == 1 and isinstance(text, str) else tuple(sequence[i:i + order])
        for i in range(max(0, len(sequence) - order + 1))
    )


def _fallback_bleu(predictions: list[str], references: list[str]) -> float:
    precisions = []
    for order in range(1, 5):
        matched = total = 0
        for prediction, reference in zip(predictions, references):
            candidate = _ngram_counts(prediction.split(), order)
            target = _ngram_counts(reference.split(), order)
            matched += sum((candidate & target).values())
            total += sum(candidate.values())
        precisions.append((matched + 1) / (total + 1))
    if not predictions:
        return 0.0
    candidate_length = sum(len(x.split()) for x in predictions)
    reference_length = sum(len(x.split()) for x in references)
    brevity_penalty = 1.0 if candidate_length > reference_length else math.exp(1 - reference_length / max(candidate_length, 1))
    return 100 * brevity_penalty * math.exp(sum(math.log(p) for p in precisions) / 4)


def _fallback_chrf(predictions: list[str], references: list[str]) -> float:
    scores = []
    for prediction, reference in zip(predictions, references):
        f_scores = []
        for order in range(1, 7):
            candidate = _ngram_counts(prediction, order)
            target = _ngram_counts(reference, order)
            overlap = sum((candidate & target).values())
            precision = overlap / max(sum(candidate.values()), 1)
            recall = overlap / max(sum(target.values()), 1)
            f_scores.append((2 * precision * recall) / max(precision + recall, 1e-12))
        scores.append(sum(f_scores) / len(f_scores))
    return 100 * sum(scores) / max(len(scores), 1)


def identity_metrics(path: str | Path) -> dict[str, float | str | int]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    predictions = [row["egyptian"] for row in rows]
    references = [row["msa"] for row in rows]
    try:
        import sacrebleu
    except ImportError:
        return {"backend": "fallback", "rows": len(rows), "bleu": _fallback_bleu(predictions, references),
                "chrf": _fallback_chrf(predictions, references)}
    return {"backend": "sacrebleu", "rows": len(rows),
            "bleu": sacrebleu.corpus_bleu(predictions, [references]).score,
            "chrf": sacrebleu.corpus_chrf(predictions, [references]).score}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/egy_msa/test.jsonl", type=Path)
    parser.add_argument("--output", default="reports/identity_baseline.json", type=Path)
    args = parser.parse_args()
    report = identity_metrics(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

