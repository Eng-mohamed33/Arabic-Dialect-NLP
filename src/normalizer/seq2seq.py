"""Load the fine-tuned Egyptian->MSA model once and reuse it across requests."""
from pathlib import Path

from src.data.text import dataset_orthography
from src.normalizer.prompt import PROMPT


class Seq2SeqNormalizer:
    def __init__(self, model_dir: str | Path, *, max_length: int = 128, num_beams: int = 4) -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self._torch = torch
        self.max_length, self.num_beams = max_length, num_beams
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir)).to(self.device).eval()

    def __call__(self, text: str) -> str:
        cleaned = dataset_orthography(text)  # training text had no ة/ى and was normalized: inputs must match
        if not cleaned:
            return ""
        inputs = self.tokenizer(PROMPT + cleaned, return_tensors="pt", truncation=True,
                                max_length=self.max_length).to(self.device)
        with self._torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=self.max_length, num_beams=self.num_beams)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
