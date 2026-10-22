import json
from pathlib import Path

from src.eval.translation import identity_metrics


def test_identity_metrics_returns_scores(tmp_path: Path):
    path = tmp_path / "test.jsonl"
    path.write_text(json.dumps({"egyptian": "انا رايح", "msa": "انا ذاهب"}, ensure_ascii=False) + "\n", encoding="utf-8")
    result = identity_metrics(path)
    assert result["rows"] == 1
    assert 0 <= result["bleu"] <= 100
    assert 0 <= result["chrf"] <= 100



def test_build_report_scores_model_and_identity():
    from src.normalizer.evaluate import build_report
    report = build_report(["انا رايح", "هو جاي"], ["انا ذاهب", "هو جاي"], ["انا ذاهب", "هو جاي"])
    assert report["non_identical_rows"] == 1
    assert report["model"]["chrf"] > report["identity"]["chrf"]
    assert report["copy_rate"] == 0.5
