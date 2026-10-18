# AGENTS.md

Greenfield project: a small text-summarization LLM app (Python). No application code exists yet — only a pre-created virtualenv. There is no git repo, no tests, no lint config.

## Environment

- Platform is Windows; default shell is PowerShell (no `&&` chaining).
- Virtualenv lives at `venv/` (not `.venv/`), Python **3.14.7** from `C:\Python314`.
  - Activate: `.\venv\Scripts\activate`
  - Interpreter: `.\venv\Scripts\python.exe`
- Only `pip` is installed so far. Do not assume any package is available.

## Planned stack (agreed with user)

- UI: **Streamlit** (`streamlit run app.py`)
- Model: **local HuggingFace** via `transformers.pipeline("summarization")`, default `t5-small`; user may later swap the model-loading function for the OpenAI API.
- Intended main entrypoint: `app.py` in the repo root.

## Gotchas

- Python 3.14 is bleeding-edge: before writing code, check that wheels exist for `torch` (and `transformers` deps) on this version. If `torch` fails to install, flag it to the user instead of working around it silently.
- Install CPU-only torch to keep the download small:
  `pip install torch --index-url https://download.pytorch.org/whl/cpu`
- HuggingFace models download to `~/.cache/huggingface` on first run; wrap model loading in `@st.cache_resource` so it loads once per session, not per rerun.
- HF summarization pipelines truncate past ~512 tokens — long input needs chunking (map-reduce), not a single call.
