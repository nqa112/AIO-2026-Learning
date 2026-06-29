import os

from huggingface_hub import InferenceClient

from src.config import EMBED_MODEL, LLM_MODEL, PROMPT, RagConfig
from src.schemas import Citation, RagResponse


class HuggingFaceClient:
    """Small wrapper for the Hugging Face Inference API."""

    def __init__(self):
        """Create the client with HF_TOKEN."""
        self.client = InferenceClient(api_key=os.environ["HF_TOKEN"])

    def embed(self, texts):
        """Create embeddings for texts."""
        return self.client.feature_extraction(texts, model=EMBED_MODEL).tolist()

    def embed_batches(self, texts, batch_size, progress_callback=None, progress_start=0, progress_end=1):
        """Embed texts in small batches to avoid large API requests."""
        embeddings = []
        total_batches = max(1, (len(texts) + batch_size - 1) // batch_size)
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            embeddings.extend(self.embed(batch))
            if progress_callback:
                batch_index = (start // batch_size) + 1
                progress = progress_start + (
                    (progress_end - progress_start) * batch_index / total_batches
                )
                progress_callback(progress, f"Embedding chunks {min(start + batch_size, len(texts))}/{len(texts)}")
        return embeddings

    def chat(self, prompt, config):
        """Send a prompt to the LLM and return the answer."""
        response = self.client.chat_completion(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=config.max_tokens,
        )
        return response.choices[0].message.content


class LLMAnswerer:
    """Build the prompt, call the LLM, and create citations."""

    def __init__(self, ai_client, config=None):
        """Set the AI client and generation settings."""
        self.ai_client = ai_client
        self.config = config or RagConfig()

    def answer(self, question, retrieved_chunks):
        """Answer using retrieved chunks only."""
        if not retrieved_chunks:
            return RagResponse("Tôi không biết dựa trên nội dung tài liệu.", [])

        context = self._format_context(retrieved_chunks)
        prompt = PROMPT.format(context=context, question=question)
        answer = self.ai_client.chat(prompt, self.config)

        # Use the same chunks from the prompt as citation sources.
        citations = [
            Citation(
                file_name=chunk.metadata["file_name"],
                page=chunk.metadata["page"],
                snippet=self._short_snippet(chunk.text),
            )
            for chunk in retrieved_chunks
        ]
        return RagResponse(answer, citations)

    def _format_context(self, chunks):
        """Format chunks into numbered context blocks."""
        blocks = []
        for idx, chunk in enumerate(chunks, start=1):
            meta = chunk.metadata
            blocks.append(f"[{idx}] {meta['file_name']} - page {meta['page']}\n{chunk.text}")
        return "\n\n".join(blocks)

    def _short_snippet(self, text, limit=220):
        """Shorten chunk text for citation display."""
        text = " ".join(text.split())
        return text if len(text) <= limit else text[: limit - 3] + "..."
