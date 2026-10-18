# 📝 Text Summarizer (LLM_app)

A small, local, CPU-only text summarization app built with **Streamlit** and
**HuggingFace Transformers**. Paste or upload text, pick a length and style,
and get an abstractive summary — no API key required.

## Install

```powershell
# CPU-only torch keeps the download small (~200 MB instead of ~2.5 GB)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Requires Python 3.10+ (tested on 3.14).

## Run

```powershell
streamlit run app.py
```

The first summarize click downloads the model (~240 MB for `t5-small`) into
`~/.cache/huggingface`; afterwards it's loaded once per server run and cached.

## Usage

- Paste text into the box (or upload a `.txt` file — it fills the box).
- Sidebar controls:
  - **Model** — `t5-small` (fast, default), `t5-base`, `facebook/bart-large-cnn` (best, slow on CPU)
  - **Style** — Paragraph / One sentence / Bullet points
  - **Summary length** — target size as a percentage of the input
- Click **Summarize**. Results show word-count stats and a download button.

## How it works

```
app.py                # Streamlit UI only
summy/
├── summarizer.py     # chunking + map-reduce orchestration + cached model loading
└── backends/
    ├── base.py       # SummarizerBackend interface
    └── hf.py         # local AutoModelForSeq2SeqLM implementation
tests/
├── test_summarizer.py     # unit tests for pure logic (no model download)
└── test_summarize_text.py # map-reduce orchestration tests (stub backend)
```

> **Note on transformers v5:** the `"summarization"` pipeline was removed in
> transformers 5.x. This app therefore drives `AutoModelForSeq2seqLM.generate()`
> directly, which still works with all the classic summarization checkpoints.

Summarization models truncate input at ~512 tokens, so long text
is handled with **map-reduce**:

1. Split into ~420-token chunks on sentence boundaries.
2. Summarize each chunk (map).
3. Repeatedly summarize the partial summaries until they fit the budget (reduce).
4. Apply a final styled pass (e.g. the T5 "summarize in one sentence" prefix).

## Tests

```powershell
python -m unittest discover tests
```

Tests cover chunking, length math, and the map-reduce orchestration (with a
stub backend); they never download a model.
