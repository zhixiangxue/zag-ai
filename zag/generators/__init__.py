"""Generator utilities for zag.

This top-level package provides LLM-based generators that operate on various
input formats including strings, dicts, and zag schemas.

Current generators:
    - AnswerGenerator: produce final answers from retrieved context items.
"""

from typing import List
from pydantic import BaseModel


class Answer(BaseModel):
    """Final answer produced by generators."""

    text: str
    citations: List[str]


from .general import AnswerGenerator as GeneralAnswerGenerator

__all__ = [
    "Answer",
    "GeneralAnswerGenerator",
]
