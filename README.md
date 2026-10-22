# Arabic Dialect NLP Toolkit

Arabic dialect identification (Egyptian / Gulf / Levantine / Maghrebi) and Egyptian → Modern Standard Arabic (MSA) conversion,
with a FastAPI service and a Streamlit demo. Arabic documentation: [README.ar.md](README.ar.md).

```
text → normalize → dialect identifier → Egyptian and confident? ──yes──▶ mT5 Egyptian→MSA ──▶ MSA
                                              └─no──▶ dialect + confidence only
```

## Results

All numbers are produced by scripts in this repo (`python -m src.eval.summarize` collects them into `reports/RESULTS.md`).

**Dialect identification** — IADD after cleaning (123,009 sentences, 4 classes), stratified 80/10/10 split, seed 42, test = 12,301 sentences.

| Model | Macro-F1 | Accuracy |
|---|---|---|
| TF-IDF char 1–5-grams + LinearSVC | 0.838 | 0.938 |
| AraBERT / MARBERT / CAMeLBERT | _pending: run `notebooks/colab_dialect_transformers.ipynb`_ | |

**Egyptian → MSA** — held-out test of 2,223 sentence pairs (EGY_MSA_Translation), sacrebleu, single reference.

| System | BLEU | chrF |
|---|---|---|
| Identity (copy the input) | 16.44 | 46.98 |
| mT5-small fine-tuned (3 epochs) | **23.68** | **52.77** |

On the hand-written gold set: _pending_ (`python -m src.normalizer.evaluate --test data/gold/egy_msa_gold.csv --output reports/mt5_gold.json`).
A heuristic error analysis of the mT5 outputs is in [`reports/error_analysis.md`](reports/error_analysis.md).

## Quick start

```powershell
python -m pip install -r requirements.txt
python -m src.data.prepare                      # clean IADD + EGY→MSA files under data/raw/
python -m src.normalizer.prepare
python -m src.dialect_id.train_eval --input data/processed/dialect.csv --output-dir reports/calibrated --calibrated
python -m uvicorn app.main:app --reload         # http://127.0.0.1:8000/docs
python -m pip install -r requirements-demo.txt; python -m streamlit run app/streamlit_app.py
```

The fine-tuned model lives in `models/egy_msa_mt5` (not in the repo, 1.2 GB). Train it with `notebooks/colab_egy_msa.ipynb`.
`--calibrated` writes `threshold.json`, which the API and demo load automatically. Deployment: [docs/DEPLOY.md](docs/DEPLOY.md).

## Data (not redistributed)

| Dataset | Used for | Notes |
|---|---|---|
| [IADD](https://github.com/JihadZa/IADD) | dialect identification | CC-BY-4.0 per its Hugging Face mirror; check the original repository |
| [EGY_MSA_Translation](https://github.com/mohamedatta93/EGY_MSA_Translation) (Faheem et al., 2024) | Egyptian → MSA | no license stated: use for research, do not redistribute |

`data/` and `models/` are git-ignored. Cite the original papers if you use this work.

## Known limitations

- **Source confound**: in IADD, Egyptian and Gulf come almost entirely from news comments (AOC), Levantine mostly from SHAMI, Maghrebi from PADIC/TSAC, so a classifier can partly learn the source style. Treat accuracy with caution.
- **Egyptian is rare (about 3.7%)**: recall is 0.61 for the baseline, and short ambiguous sentences are often labeled Levantine, so `/pipeline` may not convert them (`/normalize` always does).
- **No MSA or Iraqi class**: MSA input is assigned to one of the four dialects.
- **Normalized orthography**: the parallel data has no hamza, ة or ى (e.g. "الي", "المنزل" written with ه), so model output looks the same.
- **Noisy references and a single seed**: some test references are free paraphrases; results are one run on a split from the same sources as training.
- The mT5-small model sometimes drops negation or leaves dialect words untranslated (see the error analysis).

## Layout

```
src/data         cleaning, splits, gold-set checks      src/dialect_id   baseline, calibration, transformers
src/normalizer   prepare, train, evaluate, inference    src/eval         metrics, error analysis, results summary
app              FastAPI service + Streamlit demo       notebooks        Colab notebooks
```
