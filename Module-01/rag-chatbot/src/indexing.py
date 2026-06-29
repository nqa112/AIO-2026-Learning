import hashlib
import time

import chromadb
import pymupdf

from src.config import RagConfig
from src.schemas import IndexingReport, Notebook, SourceChunk


class PDFIndexer:
    """Read PDFs, create chunks, embed them, and save them to ChromaDB."""

    def __init__(self, ai_client, config=None):
        """Set the AI client and indexing settings."""
        self.ai_client = ai_client
        self.config = config or RagConfig()

    def build_notebook(self, uploaded_files, progress_callback=None):
        """Create one notebook from the uploaded PDF files."""
        chunks, page_count, empty_page_count = self._extract_chunks(
            uploaded_files,
            progress_callback,
        )
        if not chunks:
            raise ValueError("Could not extract readable text from the uploaded PDFs.")

        collection = chromadb.Client().get_or_create_collection(
            f"notebook_{time.time_ns()}",
            configuration={"hnsw": {"space": "cosine"}},
        )
        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        embeddings = self.ai_client.embed_batches(
            documents,
            self.config.embedding_batch_size,
            progress_callback=progress_callback,
            progress_start=0.6,
            progress_end=0.95,
        )
        self._update_progress(progress_callback, 0.98, "Saving vector index")
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        self._update_progress(progress_callback, 1.0, "Notebook indexed")

        source_names = list(dict.fromkeys(uploaded_file.name for uploaded_file in uploaded_files))
        report = self._build_report(
            source_count=len(source_names),
            page_count=page_count,
            empty_page_count=empty_page_count,
            chunks=chunks,
        )
        return Notebook(
            collection=collection,
            source_names=source_names,
            report=report,
        )

    def _extract_chunks(self, uploaded_files, progress_callback=None):
        """Read each PDF page and convert text into chunks."""
        chunks = []
        page_count = 0
        empty_page_count = 0
        total_files = max(1, len(uploaded_files))

        for source_idx, uploaded_file in enumerate(uploaded_files):
            file_bytes = uploaded_file.getvalue()
            source_hash = hashlib.sha1(file_bytes).hexdigest()[:10]
            source_id = f"{source_idx}_{source_hash}"
            chunk_index = 0

            with pymupdf.open(stream=file_bytes, filetype="pdf") as reader:
                for page_index, page in enumerate(reader, start=1):
                    page_count += 1
                    page_text = page.get_text("text", sort=True) or ""
                    if not page_text.strip():
                        empty_page_count += 1

                    for text in self._chunk_text(page_text):
                        # Keep source info so citations can show file name and page.
                        metadata = {
                            "source_id": source_id,
                            "file_name": uploaded_file.name,
                            "page": page_index,
                            "chunk_index": chunk_index,
                        }
                        chunks.append(
                            SourceChunk(
                                text=text,
                                metadata=metadata,
                                chunk_id=f"{source_id}_{page_index}_{chunk_index}",
                            )
                        )
                        chunk_index += 1

            progress = 0.6 * ((source_idx + 1) / total_files)
            self._update_progress(progress_callback, progress, "Reading and chunking PDFs")

        return chunks, page_count, empty_page_count

    def _update_progress(self, progress_callback, progress, message):
        """Update the progress bar if the UI provides a callback."""
        if progress_callback:
            progress_callback(progress, message)

    def _build_report(self, source_count, page_count, empty_page_count, chunks):
        """Create a short report after indexing finishes."""
        return IndexingReport(
            source_count=source_count,
            page_count=page_count,
            chunk_count=len(chunks),
            empty_page_count=empty_page_count,
        )

    def _chunk_text(self, text):
        """Split text into chunks with overlap between nearby chunks."""
        paras = [p.strip() for p in text.split("\n") if p.strip()]
        chunks, cur = [], ""
        step = max(1, self.config.chunk_size - self.config.chunk_overlap)

        for para in paras:
            if len(para) > self.config.chunk_size:
                if cur.strip():
                    chunks.append(cur.strip())
                    cur = ""
                for start in range(0, len(para), step):
                    piece = para[start : start + self.config.chunk_size].strip()
                    if piece:
                        chunks.append(piece)
                continue

            if len(cur) + len(para) + 1 <= self.config.chunk_size:
                cur += para + "\n"
            else:
                if cur.strip():
                    chunks.append(cur.strip())
                cur = (
                    cur[-self.config.chunk_overlap :] + para + "\n"
                    if self.config.chunk_overlap
                    else para + "\n"
                )

        if cur.strip():
            chunks.append(cur.strip())

        return chunks
