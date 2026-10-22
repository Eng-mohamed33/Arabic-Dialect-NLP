"""Run inference with a fine-tuned Egyptian-to-MSA model."""

import argparse

from src.normalizer.seq2seq import Seq2SeqNormalizer


def normalize(text: str, model_dir: str, *, max_length: int = 128) -> str:
    return Seq2SeqNormalizer(model_dir, max_length=max_length)(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models/egy_msa_mt5")
    parser.add_argument("--text", required=True)
    args = parser.parse_args()
    print(normalize(args.text, args.model_dir))


if __name__ == "__main__":
    main()
