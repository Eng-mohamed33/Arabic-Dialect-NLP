"""Fine-tune an Arabic BERT-style encoder for dialect identification (run on a GPU, e.g. Colab).

    python -m src.dialect_id.train_transformer --model marbert --input data/processed/dialect.csv

Uses the same stratified split and seed as the TF-IDF baseline, so macro-F1 values are directly comparable.
Only metrics are written by default (no checkpoints), which keeps Drive usage tiny.
"""
import argparse
import tempfile
from pathlib import Path

from src.data.load import load_dataset
from src.data.split import stratified_split
from src.eval.metrics import classification_metrics, save_report

MODELS = {
    "arabert": "aubmindlab/bert-base-arabertv02",
    "marbert": "UBC-NLP/MARBERTv2",
    "camelbert": "CAMeL-Lab/bert-base-arabic-camelbert-da",
}
ALLOWED_LABELS = ["Egyptian", "Gulf", "Levantine", "Maghrebi", "MSA", "Iraqi"]


def present_labels(labels) -> list[str]:
    seen = set(labels)
    return [label for label in ALLOWED_LABELS if label in seen]


def cap_per_class(frame, max_per_class: int | None, seed: int):
    """Optionally cap every class (training split only) to cut training time."""
    if not max_per_class:
        return frame
    parts = [group.sample(n=min(len(group), max_per_class), random_state=seed) for _, group in frame.groupby("label")]
    import pandas as pd
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def run(model: str, input_path: str | Path, output_dir: str | Path, *, seed: int = 42, epochs: float = 3,
        batch_size: int = 32, learning_rate: float = 2e-5, max_length: int = 64,
        max_per_class: int | None = None, save_model: bool = False) -> dict:
    import torch
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                              Trainer, TrainingArguments)

    model_id = MODELS.get(model, model)
    frame = load_dataset(input_path, labels=ALLOWED_LABELS)
    labels = present_labels(frame["label"])
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    train_frame, validation_frame, test_frame = stratified_split(frame, seed=seed)  # same split as the baseline
    train_frame = cap_per_class(train_frame, max_per_class, seed)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    network = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=len(labels), id2label=id2label, label2id=label2id)

    class Encoded(torch.utils.data.Dataset):
        def __init__(self, part):
            self.encodings = tokenizer(part["text"].tolist(), truncation=True, max_length=max_length)
            self.labels = [label2id[label] for label in part["label"]]

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, index):
            item = {key: values[index] for key, values in self.encodings.items()}
            item["labels"] = self.labels[index]
            return item

    args = TrainingArguments(
        output_dir=tempfile.mkdtemp(prefix="dialect_ckpt_"), per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2, learning_rate=learning_rate, num_train_epochs=epochs,
        weight_decay=0.01, eval_strategy="epoch", save_strategy="no", logging_steps=500, report_to="none",
        seed=seed, fp16=torch.cuda.is_available(), dataloader_num_workers=0)
    trainer = Trainer(model=network, args=args, train_dataset=Encoded(train_frame), eval_dataset=Encoded(validation_frame),
                      data_collator=DataCollatorWithPadding(tokenizer), processing_class=tokenizer)
    trainer.train()

    logits = trainer.predict(Encoded(test_frame)).predictions
    predictions = [id2label[int(i)] for i in logits.argmax(axis=-1)]
    report = classification_metrics(test_frame["label"], predictions, labels)
    report.update({"model": model_id, "seed": seed, "epochs": epochs, "learning_rate": learning_rate,
                   "max_length": max_length, "max_per_class": max_per_class,
                   "sizes": {"all": len(frame), "train": len(train_frame), "validation": len(validation_frame),
                             "test": len(test_frame)}})
    destination = Path(output_dir) / model.replace("/", "_")
    save_report(report, destination)
    if save_model:
        trainer.save_model(str(destination / "model"))
        tokenizer.save_pretrained(str(destination / "model"))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="marbert", help=f"one of {sorted(MODELS)} or any Hugging Face model id")
    parser.add_argument("--input", default="data/processed/dialect.csv", type=Path)
    parser.add_argument("--output-dir", default="reports/transformers", type=Path)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--epochs", default=3, type=float)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--max-length", default=64, type=int)
    parser.add_argument("--max-per-class", default=None, type=int,
                        help="cap training rows per class (e.g. 25000) to shorten training")
    parser.add_argument("--save-model", action="store_true")
    args = parser.parse_args()
    report = run(args.model, args.input, args.output_dir, seed=args.seed, epochs=args.epochs,
                 batch_size=args.batch_size, learning_rate=args.lr, max_length=args.max_length,
                 max_per_class=args.max_per_class, save_model=args.save_model)
    print(f"{args.model}: macro_f1={report['macro_f1']:.4f} accuracy={report['accuracy']:.4f}")


if __name__ == "__main__":
    main()
