"""Model evaluation and comparison."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.evaluation.evaluator import Evaluator

__all__ = ["Evaluator"]


def __getattr__(name: str):
    if name == "Evaluator":
        from src.evaluation.evaluator import Evaluator

        return Evaluator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
