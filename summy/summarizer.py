"""Model loading, chunking, and the map-reduce orchestration.

The pure helpers (``estimate_tokens``, ``split_into_chunks``,
``length_bounds``) are side-effect free and unit-testable without
downloading a model.
"""

from __future__ import annotations

import math
import re

import streamlit as st

from .backends.base import SummarizerBackend
from .backends.hf import HFBackend

# --- Pure helpers (no ML, no Streamlit) -------------------------------------

#: Conservative words→tokens ratio for English text.
TOKENS_PER_WORD = 1.3
#: Stay well under the 512-token limit of t5/bart encoders.
CHUNK_TOKEN_BUDGET = 420
#: Minimum input length (words) worth summarizing.
MIN_INPUT_WORDS = 30

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def estimate_tokens(text: str) -> int:
    """Rough token estimate for *text* (good enough for chunk sizing)."""
    return int(len(text.split()) * TOKENS_PER_WORD)


def length_bounds(word_count: int, percent: int) -> tuple[int, int]:
    """Turn a "percent of input" target into (min_length, max_length)."""
    max_length = max(20, int(word_count * percent / 100))
    min_length = max(10, int(max_length * 0.5))
    return min_length, max_length


def _pack_words(words: list[str], max_tokens: int) -> list[str]:
    """Group words into chunks, assuming the worst case of 2 tokens/word."""
    packs: list[str] = []
    current: list[str] = []
    tokens = 0
    for word in words:
        if current and tokens + 2 > max_tokens:
            packs.append(" ".join(current))
            current, tokens = [], 0
        current.append(word)
        tokens += 2
    if current:
        packs.append(" ".join(current))
    return packs


def split_into_chunks(text: str, max_tokens: int = CHUNK_TOKEN_BUDGET) -> list[str]:
    """Split *text* on sentence boundaries into ~*max_tokens* chunks.

    Sentences longer than the budget themselves (minified JSON, bad OCR)
    are hard-split on word boundaries.
    """
    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sentence in _SENTENCE_RE.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        # Round up so accumulated chunks never exceed the budget.
        sentence_tokens = math.ceil(len(sentence.split()) * TOKENS_PER_WORD)
        if sentence_tokens > max_tokens:
            # Flush what we have, then hard-split the oversized sentence.
            if current:
                chunks.append(" ".join(current))
                current, current_tokens = [], 0
            chunks.extend(_pack_words(sentence.split(), max_tokens))
            continue
        if current and current_tokens + sentence_tokens > max_tokens:
            chunks.append(" ".join(current))
            current, current_tokens = [], 0
        current.append(sentence)
        current_tokens += sentence_tokens

    if current:
        chunks.append(" ".join(current))
    return chunks


# --- App-level orchestration -------------------------------------------------

STYLE_PREFIXES = {
    "Paragraph": "",
    "One sentence": "summarize in one sentence: ",
    "Bullet points": "summarize as bullet points: ",
}


def summarize_text(
    text: str,
    backend: SummarizerBackend,
    *,
    percent: int = 30,
    style: str = "Paragraph",
    max_tokens: int = CHUNK_TOKEN_BUDGET,
) -> str:
    """Summarize *text* with map-reduce chunking.

    Short inputs go through the backend in one call. Long inputs are
    chunked, each chunk summarized (map), the partial summaries reduced
    until they fit the budget, and a final styled pass is applied.

    The final pass targets ``percent`` of the *original* input length, so the
    requested length is honoured regardless of how many chunk/reduce passes
    the text went through. Without this, every extra pass compounded the
    compression and a longer input produced a *shorter* summary.
    """
    chunks = split_into_chunks(text, max_tokens)
    if not chunks:
        return ""

    # The length we ultimately want, in words. Derived from the whole input,
    # not from the intermediate summaries, so chunking never changes the goal.
    target_min, target_max = length_bounds(len(text.split()), percent)

    if len(chunks) == 1:
        joined = chunks[0]
    else:
        # Map stage: summarize each chunk. Each chunk keeps only what matters,
        # so aim well below the final target to leave room for the reduce stage.
        partials = []
        for chunk in chunks:
            mn, mx = length_bounds(len(chunk.split()), max(percent, 40))
            partials.append(
                backend.summarize(chunk, min_length=mn, max_length=mx, style=style)
            )
        joined = " ".join(partials)
        # Reduce only when the mapped text genuinely won't fit one call. Each
        # pass halves the text, so looping to the chunk budget would drive
        # ``joined`` down to ~420 tokens regardless of ``percent`` and the final
        # pass could never reach the requested length. The backend truncates its
        # own oversized input, so a little overflow is safe — and this bound
        # makes the reduction total, so the loop always terminates.
        while estimate_tokens(joined) > max_tokens * 2:
            reduced = []
            for piece in split_into_chunks(joined, max_tokens):
                mn, mx = length_bounds(len(piece.split()), 50)
                reduced.append(
                    backend.summarize(
                        piece, min_length=mn, max_length=mx, style=style
                    )
                )
            joined = " ".join(reduced)

    # Final pass: expand/condense ``joined`` to the target for the *whole*
    # input. ``style`` is passed down so each backend renders it its own way.
    # Only the encoder's *input* window is a hard limit, and that is enforced
    # inside the backend (it truncates its own input); ``max_length`` here is a
    # pure output budget, so it must not be clamped to the chunk size — doing so
    # capped every long summary to ~420 words no matter the requested length.
    mx = max(24, target_max)
    mn = min(target_min, mx - 1)
    return backend.summarize(joined, min_length=mn, max_length=mx, style=style)


def get_backend(model_name: str = "t5-small") -> SummarizerBackend:
    """Return a process-wide cached backend (loads the model once)."""
    return _cached_hf(model_name)


@st.cache_resource(show_spinner="Loading model (first run downloads it)…")
def _cached_hf(model_name: str) -> HFBackend:
    return HFBackend(model_name)
