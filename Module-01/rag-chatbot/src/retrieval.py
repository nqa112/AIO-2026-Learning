import math
import re
from collections import Counter

from src.config import RagConfig
from src.schemas import RetrievedChunk


class Retriever:
    """Find the best document chunks for a user question."""

    def __init__(self, ai_client, config=None):
        """Set the AI client and retrieval settings."""
        self.ai_client = ai_client
        self.config = config or RagConfig()

    def retrieve(self, question, notebook):
        """Search ChromaDB and return the final chunks for the LLM."""
        if notebook.collection.count() == 0:
            return []

        query_embedding = self.ai_client.embed([question])[0]
        n_results = min(self.config.search_k, notebook.collection.count())
        result = notebook.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances", "embeddings"],
        )

        candidates = self._build_candidates(question, result)
        if self._is_weak_retrieval(candidates):
            return []

        return self._select_mmr(candidates)

    def _build_candidates(self, question, result):
        """Turn Chroma results into chunks with ranking scores."""
        docs = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]
        embeddings = result["embeddings"][0]
        query_terms = self._keywords(question)
        doc_terms_list = [self._keywords(doc) for doc in docs]
        bm25_scores = self._bm25_scores(query_terms, doc_terms_list)

        candidates = []
        for doc, metadata, distance, embedding, bm25_score in zip(
            docs, metadatas, distances, embeddings, bm25_scores
        ):
            semantic_score = 1 - distance
            # BM25 helps exact terms like model names, APIs, and formulas.
            rank_score = semantic_score + (self.config.keyword_weight * bm25_score)
            candidates.append(
                RetrievedChunk(
                    text=doc,
                    metadata=metadata,
                    embedding=embedding,
                    distance=distance,
                    semantic_score=semantic_score,
                    rank_score=rank_score,
                )
            )

        return sorted(candidates, key=lambda chunk: chunk.rank_score, reverse=True)

    def _is_weak_retrieval(self, candidates):
        """Check if the retrieved chunks are relevant enough."""
        if not candidates:
            return True

        # If the best chunk is still too far, avoid guessing.
        best_distance = min(chunk.distance for chunk in candidates)
        return best_distance > self.config.max_distance

    def _select_mmr(self, candidates):
        """Use MMR to keep relevant chunks and reduce duplicates."""
        selected = []
        pool = candidates[:]

        while pool and len(selected) < self.config.final_k:
            if not selected:
                selected.append(pool.pop(0))
                continue

            best_item, best_score = None, -float("inf")
            for item in pool:
                # Penalize chunks that are too similar to already selected chunks.
                diversity_penalty = max(
                    self._cosine_similarity(item.embedding, chosen.embedding)
                    for chosen in selected
                )
                mmr_score = (self.config.mmr_lambda * item.rank_score) - (
                    (1 - self.config.mmr_lambda) * diversity_penalty
                )
                if mmr_score > best_score:
                    best_item, best_score = item, mmr_score

            selected.append(best_item)
            pool.remove(best_item)

        return selected

    def _keywords(self, text):
        """Extract normalized tokens from text."""
        return [token for token in re.findall(r"\w+", text.lower()) if len(token) > 2]

    def _bm25_scores(self, query_terms, docs_terms, k1=1.5, b=0.75):
        """Calculate BM25 scores for each document."""
        if not query_terms or not docs_terms:
            return [0.0 for _ in docs_terms]

        doc_count = len(docs_terms)
        avg_doc_len = sum(len(doc_terms) for doc_terms in docs_terms) / max(doc_count, 1)
        query_terms = set(query_terms)
        term_doc_freq = {
            term: sum(1 for doc_terms in docs_terms if term in doc_terms)
            for term in query_terms
        }
        docs_term_freq = [Counter(doc_terms) for doc_terms in docs_terms]

        scores = []
        for doc_terms, term_freq in zip(docs_terms, docs_term_freq):
            score = 0.0
            doc_len = len(doc_terms)
            length_norm = k1 * (1 - b + (b * doc_len / max(avg_doc_len, 1)))

            for term in query_terms:
                tf = term_freq.get(term, 0)
                if tf == 0:
                    continue
                df = term_doc_freq[term]
                idf = math.log(1 + ((doc_count - df + 0.5) / (df + 0.5)))
                score += idf * ((tf * (k1 + 1)) / (tf + length_norm))

            scores.append(score)

        max_score = max(scores, default=0.0)
        if max_score > 0:
            return [score / max_score for score in scores]
        return scores

    def _cosine_similarity(self, vec_a, vec_b):
        """Calculate cosine similarity between two vectors."""
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if not norm_a or not norm_b:
            return 0
        return dot / (norm_a * norm_b)
