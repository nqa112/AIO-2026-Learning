import json
import os
from dataclasses import dataclass
from pathlib import Path


LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
CONFIG_PATH = Path(__file__).resolve().parent.parent / "rag_config.json"


@dataclass
class RagConfig:
    """Store settings for the RAG pipeline."""

    chunk_size: int = 1000
    chunk_overlap: int = 180
    search_k: int = 12
    final_k: int = 4
    max_distance: float = 0.5
    keyword_weight: float = 0.08
    mmr_lambda: float = 0.75
    embedding_batch_size: int = 16
    max_tokens: int = 1024


def load_config(path=CONFIG_PATH):
    """Load settings from JSON, or use defaults if the file is missing."""
    config = RagConfig()
    if not path.exists():
        return config

    with path.open("r", encoding="utf-8") as f:
        values = json.load(f)

    allowed_fields = set(config.__dataclass_fields__)
    clean_values = {k: v for k, v in values.items() if k in allowed_fields}
    return RagConfig(**{**config.__dict__, **clean_values})


PROMPT = """You are a document-grounded QA assistant.
Use only the provided context to answer. If the context is insufficient, say that you do not know.
Answer concisely and accurately in Vietnamese. When using context, refer to sources with [1], [2], etc.

Context:
{context}

Question: {question}

Answer:"""
