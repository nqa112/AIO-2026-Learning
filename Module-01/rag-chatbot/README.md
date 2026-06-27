# PDF RAG Chatbot

A simple Streamlit chatbot that answers questions from an uploaded PDF using retrieval-augmented generation (RAG) and Hugging Face serverless inference.

**Live demo:** [aio2026-rag.streamlit.app](https://aio2026-rag.streamlit.app/)

## Key Features

- Extracts and splits text from PDF files
- Stores document embeddings in an in-memory ChromaDB collection
- Retrieves the four most relevant chunks for each question
- Rejects unrelated questions using a cosine-distance threshold
- Preserves the document and chat history with Streamlit session state

## Models

- **Chat model:** `Qwen/Qwen2.5-7B-Instruct`
- **Embedding model:** `BAAI/bge-m3`

Both models run through the Hugging Face Inference API, so no local model server is required.

## Files

- `rag.py`: PDF extraction, text chunking, embeddings, retrieval, and answer generation
- `chatbot_app.py`: Streamlit interface, file upload, and chat history
- `requirements.txt`: Python dependencies

## Local Setup

```bash
# Create the environment with Python 3.12
conda create -n rag_chatbot python=3.12 -y

# Activate the environment
conda activate rag_chatbot

# Install dependencies
pip install -r requirements.txt

# Set your Hugging Face access token
export HF_TOKEN="your_hugging_face_token"

# Start the app
streamlit run chatbot_app.py
```

Open the local URL shown in the terminal. Upload a PDF and click **Xử lý PDF** before asking questions.
