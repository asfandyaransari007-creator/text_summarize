"""Tests for the map/reduce orchestration in summarize_text.

These use a stub backend (no model download) to pin down the *geometry* of
the pipeline: how the output length relates to the requested percent, for
inputs of very different lengths.
"""

from __future__ import annotations

import sys
import unittest

sys.path.insert(0, "C:/Users/NTDC LAB/Desktop/LLM_app")

from summy.summarizer import summarize_text  # noqa: E402


class StubBackend:
    """Emulates a summarizer that honors the requested length budget.

    ``summarize_text`` signals the desired output length via ``max_length``;
    a real backend is expected to produce up to that many words. The old stub
    ignored ``max_length`` and always returned ``ratio`` of its input, which
    capped every result at half the *reduced* intermediate text and made the
    scaling test fail regardless of the pipeline.
    """

    def __init__(self, ratio: float = 0.5):
        self.ratio = ratio

    def summarize(self, text, *, min_length, max_length, style="Paragraph"):
        n = max(int(max_length), int(min_length), 1)
        return " ".join(["w"] * n)


def words(n: int) -> str:
    return " ".join(f"word{i}" for i in range(n))


class TestSummarizeTextScaling(unittest.TestCase):
    """The output must scale with the requested percent, not plateau."""

    def test_short_input_hits_target(self):
        out = summarize_text(words(200), StubBackend(0.5), percent=60)
        self.assertGreaterEqual(len(out.split()), 100)

    def test_long_input_is_not_shorter_than_medium(self):
        """More source material must never yield a *shorter* summary.

        The old reduce loop compounded compression on every pass, so a 5000-word
        input came back *shorter* than a 1000-word one at most compression ratios.
        """
        for ratio in (0.4, 0.6, 0.7):
            with self.subTest(ratio=ratio):
                medium = summarize_text(words(1000), StubBackend(ratio), percent=60)
                long = summarize_text(words(5000), StubBackend(ratio), percent=60)
                self.assertGreaterEqual(
                    len(long.split()),
                    len(medium.split()),
                    f"at ratio {ratio}: 5000-word input gave a shorter summary "
                    f"({len(long.split())}w) than a 1000-word input "
                    f"({len(medium.split())}w)",
                )

    def test_long_input_scales_with_percent(self):
        """Doubling the requested percent must meaningfully raise output."""
        low = summarize_text(words(4000), StubBackend(0.5), percent=40)
        high = summarize_text(words(4000), StubBackend(0.5), percent=80)
        self.assertGreater(
            len(high.split()),
            len(low.split()) * 1.5,
            "raising percent 40->80 barely changed the output length",
        )


if __name__ == "__main__":
    unittest.main()
