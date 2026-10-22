# Deploying the demo (Hugging Face Spaces)

1. **Create a model repo on the Hub** (private is fine) and upload the models with this layout:

```
egy_msa_mt5/            <- contents of models/egy_msa_mt5 (config.json, model.safetensors, tokenizer files)
dialect/dialect_baseline.joblib
dialect/threshold.json  <- only if you trained with --calibrated
```

```python
from huggingface_hub import HfApi
api = HfApi()  # run `python -m huggingface_hub login` first, or set HF_TOKEN
api.create_repo("YOUR_USER/arabic-dialect-models", repo_type="model", private=True, exist_ok=True)
api.upload_folder(folder_path="models/egy_msa_mt5", repo_id="YOUR_USER/arabic-dialect-models", path_in_repo="egy_msa_mt5")
api.upload_file(path_or_fileobj="reports/calibrated/dialect_baseline.joblib", path_in_repo="dialect/dialect_baseline.joblib", repo_id="YOUR_USER/arabic-dialect-models")
api.upload_file(path_or_fileobj="reports/calibrated/threshold.json", path_in_repo="dialect/threshold.json", repo_id="YOUR_USER/arabic-dialect-models")
```

2. **Create a Space** (SDK: Streamlit) and push these files to it: `app/`, `src/`, plus `deploy/spaces/README.md`
   as the Space's `README.md` and `deploy/spaces/requirements.txt` as its `requirements.txt`.
3. In the Space **Settings → Variables and secrets**, add `HF_MODEL_REPO = YOUR_USER/arabic-dialect-models`
   (and `HF_TOKEN` as a secret if the model repo is private).
4. The first start downloads about 1.2 GB; the free CPU Space is slow, so expect a few seconds per sentence.

Do not upload `data/`. The datasets are research-only and must not be redistributed.
