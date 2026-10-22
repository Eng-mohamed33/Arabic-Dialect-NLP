"""Fine-tune a multilingual seq2seq model on prepared Egyptian/MSA pairs."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from src.normalizer.prompt import PROMPT
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments)


class ParallelDataset(Dataset):
    def __init__(self, path: str | Path, tokenizer, max_length: int = 128):
        self.rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        source = PROMPT + row["egyptian"]
        encoded = self.tokenizer(source, max_length=self.max_length, truncation=True)
        target = self.tokenizer(text_target=row["msa"], max_length=self.max_length, truncation=True)
        encoded["labels"] = target["input_ids"]
        return {key: torch.tensor(value) for key, value in encoded.items()}


def train(model_name: str, train_path: str | Path, valid_path: str | Path,
          output_dir: str | Path, *, epochs: float = 3, batch_size: int = 8,
          max_length: int = 128, learning_rate: float = 1e-4, fp16: bool = False) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    train_data = ParallelDataset(train_path, tokenizer, max_length)
    valid_data = ParallelDataset(valid_path, tokenizer, max_length)
    args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir), num_train_epochs=epochs,
        per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate, weight_decay=0.01, eval_strategy="epoch",
        save_strategy="epoch", save_total_limit=1, load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False,
        predict_with_generate=False, fp16=fp16,
        logging_steps=100, report_to="none",
    )
    trainer = Seq2SeqTrainer(
        model=model, args=args, train_dataset=train_data, eval_dataset=valid_data,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="google/mt5-small")
    parser.add_argument("--train", dest="train_path", default="data/processed/egy_msa/train.jsonl", type=Path)
    parser.add_argument("--valid", dest="valid_path", default="data/processed/egy_msa/valid.jsonl", type=Path)
    parser.add_argument("--output-dir", default="models/egy_msa_mt5", type=Path)
    parser.add_argument("--epochs", default=3, type=float)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--max-length", default=128, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--fp16", action="store_true", help="off by default: mT5 can produce NaN loss in fp16")
    args = parser.parse_args()
    train(args.model_name, args.train_path, args.valid_path, args.output_dir,
          epochs=args.epochs, batch_size=args.batch_size, max_length=args.max_length,
          learning_rate=args.lr, fp16=args.fp16)


if __name__ == "__main__":
    main()

