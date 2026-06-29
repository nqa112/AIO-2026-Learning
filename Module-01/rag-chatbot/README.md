# PDF RAG Chatbot

A Streamlit RAG chatbot for asking source-grounded questions over multiple uploaded PDFs.

**Live demo:** [aio2026-rag.streamlit.app](https://aio2026-rag.streamlit.app/)

## Plans (in progress)

- Vietnamese answers are currently supported. Language selection will be added later.
- Better retrieval with hybrid search, query rewriting, and reranking.

## Key Features

- Multi-PDF notebook indexing with page metadata.
- Configurable chunking with batched Hugging Face embeddings.
- ChromaDB vector search with keyword boost and MMR diversity.
- Source-grounded answers with concise citations and Vietnamese answer support.
- Indexing progress bar and compact indexing report.
- JSON-based RAG configuration.

## Project Structure

```text
.
├── chatbot_app.py      # Streamlit app
├── rag_config.json     # RAG settings
└── src/
    ├── indexing.py     # PDF parsing, chunking, embedding, Chroma indexing
    ├── retrieval.py    # Vector search, keyword boost, MMR selection
    ├── llm.py          # Hugging Face calls, prompt, answer, citations
    ├── service.py      # Connects indexing, retrieval, and LLM steps
    └── schemas.py      # Shared data classes
```

## Tech Stack

- **App:** Streamlit.
- **PDF parsing:** PyMuPDF.
- **Vector store:** ChromaDB.
- **LLM inference:** Hugging Face Inference API.
- **Chat model:** `Qwen/Qwen2.5-7B-Instruct`.
- **Embedding model:** `BAAI/bge-m3`.

## Run Locally

```bash
conda create -n rag_chatbot python=3.12 -y
conda activate rag_chatbot
pip install -r requirements.txt
export HF_TOKEN="your_hugging_face_token"
streamlit run chatbot_app.py
```

Chunking, retrieval, and generation configurations are in `rag_config.json`.
