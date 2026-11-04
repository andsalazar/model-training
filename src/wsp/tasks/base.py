"""Task interface.

A task turns a model into a measurable score, and knows how to present the
relevant weights to an editor. Keeping scoring separate from editing is what
lets the same task run against any editor.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field


@dataclass
class TaskResult:
    """Outcome of one task run."""

    task: str
    editor: str
    score: float
    baseline: float | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    notes: str = ""

    @property
    def delta(self) -> float | None:
        return None if self.baseline is None else self.score - self.baseline

    def as_row(self) -> dict[str, object]:
        return {
            "task": self.task,
            "editor": self.editor,
            "score": round(self.score, 4),
            "baseline": None if self.baseline is None else round(self.baseline, 4),
            "delta": None if self.delta is None else round(self.delta, 4),
            **{k: round(v, 4) for k, v in self.metrics.items()},
            "notes": self.notes,
        }


class Task(abc.ABC):
    """A reproducible experiment with a numeric score."""

    #: Short identifier used in result tables and filenames.
    name: str

    @abc.abstractmethod
    def baseline(self) -> float:
        """Score of the unmodified model."""

    @abc.abstractmethod
    def evaluate(self, weights_path: str) -> float:
        """Score a checkpoint."""
