import os
import tempfile
import time

import chromadb
import ollama
import pypdf


LLM_MODEL = "vicuna:7b-v1.5-q5_1"
EMBED_MODEL = "bge-m3"

PROMPT = """Bạn là trợ lý hỏi đáp. Dùng các đoạn ngữ cảnh dưới đây để trả lời câu hỏi.
Nếu ngữ cảnh không có thông tin, hãy nói là bạn không biết, đừng bịa.
Trả lời ngắn gọn, chính xác, bằng tiếng Việt.

Ngữ cảnh:
{context}

Câu hỏi: {question}

Trả lời:"""


# Hàm tạo embedding từ danh sách text
def embed(texts):
    return ollama.embed(model=EMBED_MODEL, input=texts)["embeddings"]


def chunk_text(text, size=1000, overlap=200):
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks, cur = [], ""

    for p in paras:
        if len(cur) + len(p) + 1 <= size:
            cur += p + "\n"
        else:
            if cur:
                chunks.append(cur.strip())
            cur = (cur[-overlap:] + p + "\n") if overlap else (p + "\n")

    if cur.strip():
        chunks.append(cur.strip())

    return chunks


def process_pdf(uploaded_file):
    # Lưu file upload thành file tạm
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        path = tmp.name

    # Đọc nội dung PDF
    try:
        reader = pypdf.PdfReader(path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    finally:
        os.unlink(path)

    # Cắt nhỏ và lưu vào ChromaDB
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("Không đọc được nội dung PDF.")

    client = chromadb.Client()
    collection = client.get_or_create_collection(f"rag_{time.time_ns()}")
    collection.add(
        ids=[str(i) for i in range(len(chunks))],
        documents=chunks,
        embeddings=embed(chunks),
    )
    return collection, len(chunks)


def rag(question, collection, k=4):
    k = min(k, collection.count())
    res = collection.query(query_embeddings=embed([question]), n_results=k)
    context = "\n\n".join(res["documents"][0])
    resp = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": PROMPT.format(context=context, question=question),
            }
        ],
        options={"temperature": 0, "num_predict": 1024},
    )
    return resp["message"]["content"]
