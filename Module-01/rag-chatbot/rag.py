import os
import tempfile
import time

import chromadb
import pypdf
from huggingface_hub import InferenceClient


LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"
EMBED_MODEL = "BAAI/bge-m3"
# Khoảng cách cosine càng nhỏ thì nội dung càng liên quan
MAX_DISTANCE = 0.5
client = InferenceClient(api_key=os.environ["HF_TOKEN"])

PROMPT = """Bạn là trợ lý hỏi đáp. Dùng các đoạn ngữ cảnh dưới đây để trả lời câu hỏi.
Nếu ngữ cảnh không có thông tin, hãy nói là bạn không biết, đừng bịa.
Trả lời ngắn gọn, chính xác, bằng tiếng Việt.

Ngữ cảnh:
{context}

Câu hỏi: {question}

Trả lời:"""


# Hàm tạo embedding từ danh sách text
def embed(texts):
    return client.feature_extraction(texts, model=EMBED_MODEL).tolist()


def chunk_text(text, size=1000, overlap=200):
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks, cur = [], ""

    for p in paras:
        if len(cur) + len(p) + 1 <= size:
            cur += p + "\n"
        else:
            if cur:
                chunks.append(cur.strip())
            # Giữ lại phần cuối để các chunk không mất ngữ cảnh
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
        # Xóa file tạm sau khi đọc xong
        os.unlink(path)

    # Cắt nhỏ và lưu vào ChromaDB
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("Không đọc được nội dung PDF.")

    client = chromadb.Client()
    collection = client.get_or_create_collection(
        f"rag_{time.time_ns()}", configuration={"hnsw": {"space": "cosine"}}
    )
    collection.add(
        ids=[str(i) for i in range(len(chunks))],
        documents=chunks,
        embeddings=embed(chunks),
    )
    return collection, len(chunks)


def rag(question, collection, k=4):
    # Không lấy nhiều kết quả hơn số chunk hiện có
    k = min(k, collection.count())
    res = collection.query(query_embeddings=embed([question]), n_results=k)

    # Khoảng cách trên 0.5 nghĩa là độ tương đồng dưới 0.5, nên từ chối
    if res["distances"][0][0] > MAX_DISTANCE:
        return "Tôi không biết dựa trên nội dung tài liệu."

    context = "\n\n".join(res["documents"][0])
    resp = client.chat_completion(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": PROMPT.format(context=context, question=question),
            }
        ],
        # Giảm tính ngẫu nhiên để câu trả lời bám sát tài liệu
        temperature=0,
        max_tokens=1024,
    )
    return resp.choices[0].message.content
