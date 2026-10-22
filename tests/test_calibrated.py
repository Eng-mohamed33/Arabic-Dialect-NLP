import pandas as pd

from src.dialect_id.calibrated import build_calibrated_model


def test_calibrated_model_exposes_probabilities():
    frame = pd.DataFrame({
        "text": ["ازيك يا صاحبي", "شلونك يا خوي", "كيفك يا زلمة", "كيف حالك"] * 2,
        "label": ["Egyptian", "Gulf", "Levantine", "MSA"] * 2,
    })
    model = build_calibrated_model(cv=2)
    model.fit(frame["text"], frame["label"])
    probabilities = model.predict_proba(["ازيك"])
    assert probabilities.shape == (1, 4)
    assert abs(float(probabilities.sum()) - 1.0) < 1e-6

