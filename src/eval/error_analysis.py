"""Error analysis of the Egyptian->MSA model from reports/mt5_test_predictions.jsonl.

    python -m src.eval.error_analysis

Categories are heuristics (sentence-level chrF and token overlap), not manual labels.
Pass --train to also measure dialect words left untranslated (needs the training split).
"""
import argparse
import json
import random
import re
import statistics as st
from collections import Counter
from pathlib import Path

import sacrebleu

NEGATION = re.compile(r"\bلا\b|\bلم\b|\bلن\b|\bليس|\bلست\b|\bما\b|\bغير\b")
END_PUNCT = re.compile(r"[.!؟?،,]\s*$")
WORDS = lambda text: re.findall(r"\w+", text)


def load(path: str | Path) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    for r in rows:
        r["chrf"] = sacrebleu.sentence_chrf(r["prediction"], [r["reference"]]).score
        r["id_chrf"] = sacrebleu.sentence_chrf(r["egyptian"], [r["reference"]]).score
        r["n_src"] = len(r["egyptian"].split())
    return rows


def dialect_vocab(train_path: str | Path, min_count: int = 20, ratio: float = 0.1) -> set[str]:
    """Words frequent on the Egyptian side of the training data but (almost) absent from its MSA side."""
    egy, msa = Counter(), Counter()
    for line in Path(train_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            egy.update(WORDS(r["egyptian"]))
            msa.update(WORDS(r["msa"]))
    return {w for w, c in egy.items() if len(w) > 1 and c >= min_count and msa.get(w, 0) <= ratio * c}


def _loop(text: str) -> bool:
    w = WORDS(text)
    if any(w[i] == w[i + 1] == w[i + 2] for i in range(len(w) - 2)):
        return True
    return any(c >= 3 for c in Counter(tuple(w[i:i + 3]) for i in range(len(w) - 2)).values())


def categories(rows: list[dict], vocab: set[str] | None = None) -> dict[str, list[dict]]:
    def divergent(r):
        p = set(WORDS(r["prediction"]))
        return len(p) >= 3 and len(p - set(WORDS(r["egyptian"])) - set(WORDS(r["reference"]))) / len(p) > 0.6

    cats = {
        "worse_than_identity": [r for r in rows if r["chrf"] < r["id_chrf"] - 1],
        "severe_failure": [r for r in rows if r["chrf"] < 20],
        "large_divergence": [r for r in rows if divergent(r)],
        "negation_lost": [r for r in rows if re.search(r"\bمش\b", r["egyptian"]) and NEGATION.search(r["reference"])
                          and not NEGATION.search(r["prediction"])],
        "repetition_loop": [r for r in rows if _loop(r["prediction"]) and not _loop(r["egyptian"])],
        "over_edit_near_identical": [r for r in rows if r["id_chrf"] >= 90 and r["chrf"] < r["id_chrf"] - 15],
    }
    if vocab:
        cats["dialect_left_in_output"] = [
            r for r in rows if (set(WORDS(r["prediction"])) & vocab) - set(WORDS(r["reference"]))]
    return cats


def analyze(rows: list[dict], vocab: set[str] | None = None) -> tuple[dict, dict]:
    n = len(rows)
    cats = categories(rows, vocab)
    buckets = {"1-5": (0, 5), "6-10": (6, 10), "11-20": (11, 20), "21+": (21, 10**6)}
    by_len = {}
    for name, (lo, hi) in buckets.items():
        group = [r for r in rows if lo <= r["n_src"] <= hi]
        if group:
            by_len[name] = {"rows": len(group), "model": st.mean(r["chrf"] for r in group),
                            "identity": st.mean(r["id_chrf"] for r in group)}
    strip = lambda t: re.sub(r"\s*[.!؟?،,]+\s*$", "", t.strip())
    refs = [strip(r["reference"]) for r in rows]
    model, ident = [strip(r["prediction"]) for r in rows], [strip(r["egyptian"]) for r in rows]
    return {
        "rows": n,
        "better": sum(r["chrf"] > r["id_chrf"] + 1 for r in rows),
        "worse": len(cats["worse_than_identity"]),
        "by_length": by_len,
        "counts": {k: len(v) for k, v in cats.items()},
        "punct_added": sum(bool(END_PUNCT.search(r["prediction"])) and not END_PUNCT.search(r["reference"]) for r in rows),
        "punct_removed": sum(not END_PUNCT.search(r["prediction"]) and bool(END_PUNCT.search(r["reference"])) for r in rows),
        "no_punct": {"model_bleu": sacrebleu.corpus_bleu(model, [refs]).score,
                     "model_chrf": sacrebleu.corpus_chrf(model, [refs]).score,
                     "identity_bleu": sacrebleu.corpus_bleu(ident, [refs]).score,
                     "identity_chrf": sacrebleu.corpus_chrf(ident, [refs]).score},
    }, cats


def render(stats: dict, cats: dict, seed: int = 7, per_category: int = 3) -> str:
    n, c = stats["rows"], stats["counts"]
    pct = lambda k: f"{c[k]} ({c[k] / n:.1%})"
    np_ = stats["no_punct"]
    titles = {
        "severe_failure": "فشل شديد (chrF للجملة أقل من 20)",
        "large_divergence": "تباعد كبير عن المصدر والمرجع (يجمع مرادفات صحيحة وهلوسة)",
        "dialect_left_in_output": "كلمات لهجة بقيت في المخرجات (مفردات المصري فقط في بيانات التدريب)",
        "negation_lost": "ضياع النفي (يعكس المعنى)",
        "repetition_loop": "حلقات تكرار",
        "over_edit_near_identical": "تعديل زائد لجمل شبه متطابقة",
    }
    rng = random.Random(seed)
    lines = [
        "# Error analysis: Egyptian → MSA (mT5-small)", "",
        f"التحليل على {n} جملة من test. التصنيفات قواعد تقريبية (chrF على مستوى الجملة وتداخل الكلمات)، وليست مراجعة يدوية كاملة.", "",
        "## ملخص", "",
        f"- النموذج أفضل من نسخ الجملة (Identity) بأكثر من 1 chrF في {stats['better']} جملة ({stats['better'] / n:.0%}) وأسوأ في {stats['worse']} ({stats['worse'] / n:.0%}).",
        f"- فشل شديد (chrF أقل من 20): {pct('severe_failure')}. ليس كله خطأ نموذج: المراجع نفسها فيها ضوضاء وإعادة صياغة حرة (انظر الملاحظات).",
        f"- تباعد كبير عن المصدر والمرجع: {pct('large_divergence')}. ضياع النفي: {pct('negation_lost')}. حلقات تكرار: {pct('repetition_loop')}.",
    ]
    if "dialect_left_in_output" in c:
        lines.append(f"- مخرجات احتفظت بكلمة لهجة غير موجودة في المرجع: {pct('dialect_left_in_output')}. "
                     "المقياس تقريبي: بعض الكلمات فصيحة صحيحة لكن المراجع تتجنبها.")
    lines += [
        f"- علامة الترقيم الأخيرة ليست سبب التفوق: النموذج أضافها حيث لا توجد في المرجع {stats['punct_added']} مرة وحذفها {stats['punct_removed']} مرة، "
        f"وبعد حذفها من الطرفين يبقى التفوق: النموذج BLEU {np_['model_bleu']:.2f} / chrF {np_['model_chrf']:.2f} مقابل Identity {np_['identity_bleu']:.2f} / {np_['identity_chrf']:.2f}.", "",
        "## الأداء حسب طول الجملة (متوسط chrF للجملة)", "", "| كلمات المصدر | عدد | النموذج | Identity |", "|---|---|---|---|",
    ]
    for name, v in stats["by_length"].items():
        lines.append(f"| {name} | {v['rows']} | {v['model']:.1f} | {v['identity']:.1f} |")
    lines += ["", "## أنواع الأخطاء مع أمثلة", ""]
    for key, title in titles.items():
        if key not in c:
            continue
        lines += [f"### {title}: {pct(key)}", ""]
        pool = list(cats[key])
        rng.shuffle(pool)
        for r in pool[:per_category]:
            lines += [f"- **المصدر:** {r['egyptian']}", f"  - المرجع: {r['reference']}", f"  - النموذج: {r['prediction']}"]
        if not pool:
            lines.append("- لا توجد حالات.")
        lines.append("")
    lines += [
        "## ملاحظات", "",
        "- المراجع فيها ضوضاء: جمل غير متطابقة مع مصدرها، أو ترجمة حرة، أو أسماء بحروف لاتينية، فبعض «الأخطاء» سببها المرجع وليس النموذج.",
        "- الداتا مطبَّعة (بدون همزات، و«ه» بدل «ة»)، فمخرجات النموذج تبدو كذلك.",
        "- تشغيلة واحدة (seed واحد) على test من نفس مصدر التدريب؛ التعميم يحتاج Gold set مكتوبًا يدويًا.", "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="reports/mt5_test_predictions.jsonl", type=Path)
    parser.add_argument("--train", default="data/processed/egy_msa/train.jsonl", type=Path)
    parser.add_argument("--output", default="reports/error_analysis.md", type=Path)
    args = parser.parse_args()
    vocab = dialect_vocab(args.train) if args.train.exists() else None
    stats, cats = analyze(load(args.predictions), vocab)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(stats, cats), encoding="utf-8")
    print(json.dumps({"rows": stats["rows"], **stats["counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
