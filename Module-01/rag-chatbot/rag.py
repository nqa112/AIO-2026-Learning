from src.config import RagConfig
from src.service import RAGService


def process_pdfs(uploaded_files, progress_callback=None):
    """Index uploaded PDFs into one notebook."""
    service = RAGService()
    return service.process_pdfs(uploaded_files, progress_callback)


def answer_question(question, notebook):
    """Answer a question using an indexed notebook."""
    service = RAGService()
    return service.answer_question(question, notebook)


def process_pdf(uploaded_file):
    """Backward-compatible wrapper for the old single-PDF API."""
    notebook = process_pdfs([uploaded_file])
    return notebook, notebook.report.chunk_count


def rag(question, notebook, k=4):
    """Backward-compatible wrapper that returns only the answer text."""
    service = RAGService(RagConfig(final_k=k))
    return service.answer_question(question, notebook).answer
