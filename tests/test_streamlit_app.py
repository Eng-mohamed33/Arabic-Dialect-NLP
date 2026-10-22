import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


def test_demo_runs_without_any_model(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # no reports/ or models/ here, so the demo must degrade gracefully
    monkeypatch.setenv("PYTHONPATH", str(__import__("pathlib").Path(__file__).resolve().parents[1]))
    app = AppTest.from_file(str(__import__("pathlib").Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"))
    app.run(timeout=30)
    assert not app.exception
    app.text_area[0].set_value("انا مش عارف اروح فين")
    app.button[0].click().run(timeout=30)
    assert not app.exception and any("No dialect model" in w.value for w in app.warning)
