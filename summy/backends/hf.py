"""Local HuggingFace seq2seq backend — runs on CPU.

``transformers`` v5 removed the ``"summarization"`` pipeline, so this
backend drives ``AutoModelForSeq2SeqLM.generate()`` directly. Works with
any encoder-decoder summarization checkpoint: ``t5-small`` (fast default),
``t5-base``, ``facebook/bart-large-cnn``.
"""

from __future__ import annotations

from .base import SummarizerBackend

#: English text averages ~1.3 tokens per word for T5/BART tokenizers. Length
#: budgets arrive from the orchestrator in *words*; generation counts
#: *tokens*, so we scale by this to hit the requested word count. Without it
#: the output comes out ~25% short.
TOKENS_PER_WORD = 1.3

#: Words per sentence when translating a word budget into a sentence count.
#: Matches the natural sentence length of T5/BART summarization output.
WORDS_PER_SENTENCE = 20

def _sentence_count_for(max_length: int) -> int:
    """Pick a sentence count to ask for, from a word budget.

    Encoder-decoder summarizers emit a fixed ~3 sentences no matter how big
    the token budget is, so length only scales if the instruction names a
    sentence count. ~20 words per sentence mirrors their natural output.
    """
    return max(3, int(max_length / WORDS_PER_SENTENCE))


class HFBackend(SummarizerBackend):
    """Summarize with a local encoder-decoder model.

    The checkpoint is downloaded on first use into ``~/.cache/huggingface``
    and kept in memory for the lifetime of the process.
    """

    # T5-style models need the task prefix; BART-family models do not.
    def __init__(
        self, model_name: str = "t5-small", device: int = -1, num_beams: int = 2
    ) -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.model_name = model_name
        self.name = f"hf:{model_name}"
        self._torch = torch
        self._device = torch.device("cpu" if device < 0 else f"cuda:{device}")
        self._num_beams = num_beams
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = (
            AutoModelForSeq2SeqLM.from_pretrained(model_name)
            .to(self._device)
            .eval()
        )

    def summarize(
        self,
        text: str,
        *,
        min_length: int,
        max_length: int,
        style: str = "Paragraph",
    ) -> str:
        text = text.strip()
        if not text:
            return ""

        model_input = self._with_prefix(text, style, max_length)
        # Length budgets arrive in words; generation counts tokens. Convert so
        # the output actually reaches the requested word count.
        #
        # This is a decoder/output budget, not an encoder limit — T5's decoder
        # has unbounded positional embeddings and BART's allows 1024 positions,
        # so there is no fixed 512-token cap on generated text. (Input length
        # is bounded separately, by the truncation below.) Clamping here to 512
        # tokens capped every summary at ~390 words no matter what was asked.
        max_new = max(24, min(int(max_length * TOKENS_PER_WORD), 2048))
        min_new = max(8, min(int(min_length * TOKENS_PER_WORD), max_new - 1))

        encoded = self._tokenizer(
            model_input,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self._device)

        with self._torch.inference_mode():
            output_ids = self._model.generate(
                **encoded,
                do_sample=False,
                num_beams=self._num_beams,
                max_new_tokens=max_new,
                min_new_tokens=min_new,
            )
        return self._tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    def _with_prefix(self, text: str, style: str, max_length: int) -> str:
        # T5 is instruction-tuned on "task: input" strings. Non-T5 models
        # (e.g. BART) were not, so they get the raw text.
        if "t5" not in self.model_name.lower():
            return text
        if style == "One sentence":
            instruction = "summarize in one sentence"
        else:
            # For paragraph/bullet output, naming a sentence count is what
            # actually controls length on these models.
            n = _sentence_count_for(max_length)
            instruction = f"summarize in {n} sentences"
            if style == "Bullet points":
                instruction = f"summarize as {n} bullet points"
        return f"{instruction}: {text}"
