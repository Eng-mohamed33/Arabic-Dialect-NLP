from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app, load_normalizer
from src.normalizer.rules import normalize_to_msa
from src.pipeline import DialectPipeline


class FakeClassifier:
    def predict(self, texts):
        return ["Egyptian" for _ in texts]

    def predict_proba(self, texts):
        return [[0.9, 0.1] for _ in texts]


def _fake(text: str) -> str:
    return f"MSA:{text}"


def test_api_uses_injected_normalizer_and_pipeline():
    app = create_app(pipeline=DialectPipeline(FakeClassifier(), normalizer=_fake), normalizer=_fake,
                     normalizer_backend="fake")
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok", "dialect_model": True, "normalizer": "fake", "threshold": 0.7}
        assert client.post("/normalize", json={"text": "ازيك"}).json() == {"msa": "MSA:ازيك", "backend": "fake"}
        assert client.post("/identify", json={"text": "ازيك"}).json()["dialect"] == "Egyptian"
        result = client.post("/pipeline", json={"text": "ازيك"}).json()
        assert result["msa"] == "MSA:ازيك" and result["confidence"] == 0.9


def test_api_without_models_falls_back_to_rules():
    with TestClient(create_app()) as client:
        assert client.get("/health").json()["normalizer"] == "rules"
        assert "error" in client.post("/identify", json={"text": "ازيك"}).json()
        assert client.post("/normalize", json={"text": "ازيك"}).json()["backend"] == "rules"


def test_load_normalizer_falls_back_when_model_dir_missing(tmp_path: Path):
    normalizer, backend = load_normalizer(tmp_path / "missing")
    assert backend == "rules" and normalizer is normalize_to_msa
