from src.config import load_config
from src.indexing import PDFIndexer
from src.llm import HuggingFaceClient, LLMAnswerer
from src.retrieval import Retriever


class RAGService:
    """Connect the three RAG steps: indexing, retrieval, and generation."""

    def __init__(self, config=None):
        """Create all pipeline components with the same settings."""
        self.config = config or load_config()
        self.ai_client = HuggingFaceClient()
        self.indexer = PDFIndexer(self.ai_client, self.config)
        self.retriever = Retriever(self.ai_client, self.config)
        self.answerer = LLMAnswerer(self.ai_client, self.config)

    def process_pdfs(self, uploaded_files, progress_callback=None):
        """Index uploaded PDFs into one notebook."""
        return self.indexer.build_notebook(uploaded_files, progress_callback)

    def answer_question(self, question, notebook):
        """Retrieve chunks and generate an answer with citations."""
        retrieved_chunks = self.retriever.retrieve(question, notebook)
        return self.answerer.answer(question, retrieved_chunks)
