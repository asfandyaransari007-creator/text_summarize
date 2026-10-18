"""Backend interface every summarization implementation must satisfy."""

from abc import ABC, abstractmethod


class SummarizerBackend(ABC):
    """Contract for a summarization backend.

    Implementations must be safe to construct lazily (model downloads /
    API clients happen in ``__init__`` or later, never at import time).
    """

    #: Short human-readable name, e.g. ``"hf:t5-small"``.
    name: str = "backend"

    @abstractmethod
    def summarize(
        self, text: str, *, min_length: int, max_length: int, style: str = "Paragraph"
    ) -> str:
        """Return a summary of *text*.

        ``min_length`` / ``max_length`` are advisory budgets in tokens —
        backends should clamp them to whatever the underlying model allows.
        ``style`` is a presentation hint (e.g. ``"Bullet points"``); each
        backend renders it in whatever way its model understands.
        """
