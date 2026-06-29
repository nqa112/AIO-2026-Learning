from dataclasses import dataclass
from typing import Any


@dataclass
class SourceChunk:
    """A PDF text chunk before it is stored in ChromaDB."""

    text: str
    metadata: dict
    chunk_id: str


@dataclass
class IndexingReport:
    """Short summary of the indexing result."""

    source_count: int
    page_count: int
    chunk_count: int
    empty_page_count: int


@dataclass
class Notebook:
    """A session notebook with its vector collection and sources."""

    collection: Any
    source_names: list[str]
    report: IndexingReport



@dataclass
class RetrievedChunk:
    """A chunk selected as context for the LLM."""

    text: str
    metadata: dict
    embedding: list[float]
    distance: float
    semantic_score: float
    rank_score: float


@dataclass
class Citation:
    """Source information shown below an answer."""

    file_name: str
    page: int
    snippet: str


@dataclass
class RagResponse:
    """Final answer and its citations."""

    answer: str
    citations: list[Citation]
