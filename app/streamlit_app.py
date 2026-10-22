"""Streamlit demo: identify the dialect and, for Egyptian, convert to MSA.

    python -m streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.components import load_components  # noqa: E402

EXAMPLES = {
    "Egyptian": "عايز اروح البيت دلوقتي",
    "Gulf": "الحين بروح الدوام",
    "Levantine": "شو عم تعمل هلأ؟",
    "Maghrebi": "واش نتا خدام دابا؟",
}

st.set_page_config(page_title="Arabic Dialect Toolkit", page_icon="🗣️")
st.markdown("<style>textarea{direction:rtl;text-align:right}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading models…")
def components():
    return load_components()


pipeline, normalizer, backend = components()
st.title("Arabic Dialect Toolkit")
st.caption("Dialect identification (Egyptian / Gulf / Levantine / Maghrebi) + Egyptian → MSA conversion")

choice = st.selectbox("Example", ["(write your own)"] + list(EXAMPLES))
text = st.text_area("Text", value=EXAMPLES.get(choice, ""), height=120)

if st.button("Analyze", type="primary") and text.strip():
    if pipeline is None:
        st.warning("No dialect model found. Run `python -m src.dialect_id.train_eval` first.")
        st.info(f"Normalizer backend: {backend}")
        st.success(normalizer(text))
    else:
        result = pipeline.run(text)
        st.subheader(result["dialect"])
        st.progress(min(max(result["confidence"], 0.0), 1.0), text=f"confidence {result['confidence']:.2f}")
        st.caption(f"Egyptian→MSA threshold: {pipeline.threshold:.2f} · normalizer: {backend}")
        if result["msa"]:
            st.success(result["msa"])
        elif result["dialect"] == pipeline.egyptian_label:
            st.info("Egyptian detected, but confidence is below the threshold, so no conversion was made.")
        else:
            st.info("Conversion is available for Egyptian only.")
