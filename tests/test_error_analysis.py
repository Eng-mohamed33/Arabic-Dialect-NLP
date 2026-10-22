from src.eval.error_analysis import _loop, analyze, categories, render


def _row(src, ref, pred):
    import sacrebleu
    return {"egyptian": src, "reference": ref, "prediction": pred, "n_src": len(src.split()),
            "chrf": sacrebleu.sentence_chrf(pred, [ref]).score, "id_chrf": sacrebleu.sentence_chrf(src, [ref]).score}


def test_loop_detection_ignores_punctuation_and_flags_real_loops():
    assert _loop("اذهب اذهب اذهب الي البيت")
    assert not _loop("هل هناك الاشياء التي تريد القيام بها او تريد القيام بها.")


def test_categories_and_report():
    rows = [_row("انا مش عارف", "لا اعرف", "لا اعرف"), _row("انا مش بحبك", "انا لا احبك", "انا احبك"),
            _row("كام تمن الرحله", "ما ثمن الرحله", "كم تمن الرحله")]
    cats = categories(rows, vocab={"تمن"})
    assert len(cats["negation_lost"]) == 1
    assert len(cats["dialect_left_in_output"]) == 1
    stats, cats = analyze(rows, {"تمن"})
    assert "Error analysis" in render(stats, cats)
