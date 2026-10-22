import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from src.pipeline import DialectPipeline


def _model(path: Path):
    model = make_pipeline(TfidfVectorizer(analyzer="char", ngram_range=(1, 2)), LogisticRegression())
    model.fit(["ازيك يا باشا", "شلونك الحين"], ["Egyptian", "Gulf"])
    joblib.dump(model, path)


def test_from_path_reads_threshold_json(tmp_path: Path):
    _model(tmp_path / "dialect_baseline.joblib")
    (tmp_path / "threshold.json").write_text(json.dumps({"threshold": 0.31, "f1": 0.9}), encoding="utf-8")
    assert DialectPipeline.from_path(tmp_path / "dialect_baseline.joblib").threshold == 0.31


def test_explicit_threshold_wins_and_default_stays(tmp_path: Path):
    _model(tmp_path / "dialect_baseline.joblib")
    assert DialectPipeline.from_path(tmp_path / "dialect_baseline.joblib").threshold == 0.7
    (tmp_path / "threshold.json").write_text(json.dumps({"threshold": 0.31}), encoding="utf-8")
    assert DialectPipeline.from_path(tmp_path / "dialect_baseline.joblib", threshold=0.9).threshold == 0.9
