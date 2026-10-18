"""Streamlit entrypoint — UI only; all ML logic lives in the ``summy`` package.

Run with:  streamlit run app.py
"""

import streamlit as st

from summy.summarizer import (
    MIN_INPUT_WORDS,
    STYLE_PREFIXES,
    get_backend,
    summarize_text,
)

st.set_page_config(page_title="Text Summarizer", page_icon="📝", layout="centered")

st.title("📝 Text Summarizer")
st.caption("Runs locally with HuggingFace — no API key needed.")

# --- Sidebar settings --------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox(
        "Model",
        options=["t5-small", "t5-base", "facebook/bart-large-cnn"],
        help="t5-small is fastest; bart-large-cnn is best but slow on CPU.",
    )
    style = st.selectbox("Style", options=list(STYLE_PREFIXES))
    percent = st.slider(
        "Summary length (% of input)",
        min_value=10,
        max_value=80,
        value=35,
        help="Share of the input's length to aim for in the output. Local models top out around 60–70 words for short inputs.",
    )

# --- Input -------------------------------------------------------------------
uploaded = st.file_uploader("Upload a .txt file (optional)", type=["txt"])
text = st.text_area("Paste your text here", height=260)

if uploaded is not None and not text.strip():
    try:
        text = uploaded.read().decode("utf-8")
    except UnicodeDecodeError:
        st.error("That file isn't valid UTF-8 text. Please paste the text instead.")
        text = ""

input_words = len(text.split())
st.caption(f"Input: {input_words:,} words · ~{int(input_words * 1.3):,} tokens")

# --- Summarize ---------------------------------------------------------------
if st.button("Summarize", type="primary", disabled=not text.strip()):
    if input_words < MIN_INPUT_WORDS:
        st.warning(f"Input is very short ({input_words} words). Paste at least ~{MIN_INPUT_WORDS} words for a useful summary.")
    else:
        try:
            with st.spinner("Summarizing… (first run may take a minute while the model downloads)"):
                backend = get_backend(model_name)
                summary = summarize_text(text, backend, percent=percent, style=style)
        except Exception as exc:  # noqa: BLE001 — surface any backend error in the UI
            st.error(f"Summarization failed: {exc}")
        else:
            st.subheader("Summary")
            st.write(summary)
            target = int(input_words * percent / 100)
            out_words = len(summary.split())
            st.caption(f"Target ~{target:,} words · got {out_words:,} words · {input_words:,} words in")
            st.download_button("Download summary", summary, file_name="summary.txt", mime="text/plain")
