"""Score a fine-tuned model on a held-out set next to the identity baseline.

    python -m src.normalizer.evaluate                                   # data/processed/egy_msa/test.jsonl
    python -m src.normalizer.evaluate --test data/gold/egy_msa_gold.csv --output reports/mt5_gold.json

Both systems are scored with the same sacrebleu backend, so the numbers are comparable. CSV (gold) files are
mapped to the training orthography (no ة / ى) before generation and scoring; test.jsonl already is.
"""
import argparse
import json
from pathlib import Path

from src.data.text import dataset_orthography
from src.normalizer.prompt import PROMPT


def load_rows(path: str | Path, *, orthography: bool | None = None) -> list[dict]:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        import pandas as pd
        rows = pd.read_csv(path).dropna(subset=["egyptian", "msa"]).to_dict("records")
        orthography = True if orthography is None else orthography
    else:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        orthography = False if orthography is None else orthography
    if orthography:
        for row in rows:
            row["egyptian"], row["msa"] = dataset_orthography(str(row["egyptian"])), dataset_orthography(str(row["msa"]))
    return rows


def score(predictions: list[str], references: list[str]) -> dict[str, float]:
    import sacrebleu
    return {"bleu": sacrebleu.corpus_bleu(predictions, [references]).score,
            "chrf": sacrebleu.corpus_chrf(predictions, [references]).score}


def generate(model_dir: str | Path, sources: list[str], *, batch_size: int = 32, max_length: int = 128,
             num_beams: int = 4) -> list[str]:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    outputs: list[str] = []
    for start in range(0, len(sources), batch_size):
        batch = [PROMPT + text for text in sources[start:start + batch_size]]
        encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            generated = model.generate(**encoded, max_new_tokens=max_length, num_beams=num_beams)
        outputs.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    return outputs


def build_report(sources: list[str], references: list[str], predictions: list[str],
                 genres: list[str] | None = None) -> dict:
    changed = [i for i, (s, r) in enumerate(zip(sources, references)) if s != r]
    pick = lambda xs: [xs[i] for i in changed]
    report = {
        "rows": len(sources),
        "model": score(predictions, references),
        "identity": score(sources, references),
        "non_identical_rows": len(changed),
        "model_non_identical": score(pick(predictions), pick(references)),
        "identity_non_identical": score(pick(sources), pick(references)),
        # a high copy rate means the model mostly echoes the input (undertrained or lr too low)
        "copy_rate": sum(p == s for p, s in zip(predictions, sources)) / max(len(sources), 1),
    }
    if genres:
        report["by_genre"] = {}
        for genre in sorted({str(g) for g in genres if str(g) != "nan"}):
            idx = [i for i, g in enumerate(genres) if str(g) == genre]
            if len(idx) >= 5:
                report["by_genre"][genre] = {
                    "rows": len(idx),
                    "model_chrf": score([predictions[i] for i in idx], [references[i] for i in idx])["chrf"],
                    "identity_chrf": score([sources[i] for i in idx], [references[i] for i in idx])["chrf"],
                }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models/egy_msa_mt5")
    parser.add_argument("--test", default="data/processed/egy_msa/test.jsonl", type=Path)
    parser.add_argument("--output", default="reports/mt5_test.json", type=Path)
    parser.add_argument("--orthography", choices=["auto", "on", "off"], default="auto",
                        help="map ة->ه and ى->ي like the training data (auto: on for CSV, off for jsonl)")
    parser.add_argument("--batch-size", default=32, type=int)
    args = parser.parse_args()
    mapping = {"auto": None, "on": True, "off": False}[args.orthography]
    rows = load_rows(args.test, orthography=mapping)
    sources, references = [r["egyptian"] for r in rows], [r["msa"] for r in rows]
    predictions = generate(args.model_dir, sources, batch_size=args.batch_size)
    if mapping or (mapping is None and args.test.suffix.lower() == ".csv"):
        predictions = [dataset_orthography(p) for p in predictions]
    report = build_report(sources, references, predictions, [r.get("genre") for r in rows] if rows and "genre" in rows[0] else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    samples = args.output.with_name(args.output.stem + "_predictions.jsonl")
    samples.write_text("\n".join(json.dumps({"egyptian": s, "reference": r, "prediction": p}, ensure_ascii=False)
                                 for s, r, p in zip(sources, references, predictions)) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
