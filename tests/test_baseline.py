import pandas as pd
import pytest

from src.dialect_id.baseline import train


def test_both_baselines_train_and_predict():
    frame = pd.DataFrame({
        "text": ["ازيك يا صاحبي", "شلونك يا خوي", "كيفك يا زلمة", "كيف حالك"],
        "label": ["Egyptian", "Gulf", "Levantine", "MSA"],
    })
    for model_type in ("linearsvc", "naive_bayes"):
        model = train(frame, model_type=model_type)
        assert len(model.predict(["ازيك" ])) == 1


def test_unknown_baseline_is_rejected():
    frame = pd.DataFrame({"text": ["واحد", "اتنين"], "label": ["a", "b"]})
    with pytest.raises(ValueError, match="model_type"):
        train(frame, model_type="unknown")

