from src.eval.threshold import select_threshold


def test_select_threshold_uses_validation_scores():
    result = select_threshold(
        ["Egyptian", "Egyptian", "MSA", "MSA"],
        [0.9, 0.8, 0.2, 0.1],
        candidates=[0.5, 0.95],
    )
    assert result == {"threshold": 0.5, "f1": 1.0}

