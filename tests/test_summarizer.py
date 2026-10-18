"""Unit tests for the pure summarizer helpers — no model download required.

Run with:  python -m unittest discover tests
"""

import unittest

from summy.summarizer import (
    CHUNK_TOKEN_BUDGET,
    estimate_tokens,
    length_bounds,
    split_into_chunks,
)


class TestEstimateTokens(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(estimate_tokens(""), 0)

    def test_scales_with_words(self):
        self.assertEqual(estimate_tokens("one two three"), int(3 * 1.3))


class TestLengthBounds(unittest.TestCase):
    def test_percent_of_input(self):
        mn, mx = length_bounds(1000, 30)
        self.assertEqual(mx, 300)
        self.assertEqual(mn, 150)

    def test_never_tiny(self):
        mn, mx = length_bounds(10, 5)
        self.assertGreaterEqual(mx, 20)
        self.assertGreaterEqual(mn, 10)

    def test_min_below_max(self):
        for words in (1, 50, 500, 10_000):
            mn, mx = length_bounds(words, 30)
            self.assertLess(mn, mx)


class TestSentenceCountFor(unittest.TestCase):
    def test_scales_with_budget(self):
        from summy.backends.hf import _sentence_count_for

        self.assertEqual(_sentence_count_for(400), 20)  # 400/20
        self.assertEqual(_sentence_count_for(200), 10)

    def test_floor_is_three(self):
        from summy.backends.hf import _sentence_count_for

        self.assertEqual(_sentence_count_for(10), 3)
        self.assertEqual(_sentence_count_for(0), 3)

    def test_ceiling_tracks_decoder_budget(self):
        """A large budget must allow a proportionally large sentence count.

        The old hard cap of 25 sentences silently truncated long summaries:
        the model stops once it has emitted the requested sentence count, no
        matter how big the token budget is.
        """
        from summy.backends.hf import _sentence_count_for

        self.assertEqual(_sentence_count_for(1000), 50)
        self.assertEqual(_sentence_count_for(2000), 100)


class TestSplitIntoChunks(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(split_into_chunks(""), [])
        self.assertEqual(split_into_chunks("   \n  "), [])

    def test_short_text_single_chunk(self):
        text = "This is a sentence. This is another one."
        self.assertEqual(split_into_chunks(text), [text])

    def test_long_text_multiple_chunks_within_budget(self):
        sentence = "The quick brown fox jumps over the lazy dog near the river bank. "
        text = sentence * 200  # ~2,600 words
        chunks = split_into_chunks(text)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(estimate_tokens(chunk), CHUNK_TOKEN_BUDGET)
        # Nothing is lost in chunking.
        self.assertEqual(" ".join(chunks), text.strip())

    def test_oversized_sentence_is_hard_split(self):
        # A "sentence" with no punctuation that alone exceeds the budget.
        text = "word " * 1000
        chunks = split_into_chunks(text)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(estimate_tokens(chunk), CHUNK_TOKEN_BUDGET)

    def test_respects_sentence_boundaries(self):
        sentences = [f"Sentence number {i} is here." for i in range(30)]
        chunks = split_into_chunks(" ".join(sentences), max_tokens=100)
        for chunk in chunks:
            self.assertTrue(chunk.startswith("Sentence number"))
            self.assertTrue(chunk.endswith("is here."))


if __name__ == "__main__":
    unittest.main()
