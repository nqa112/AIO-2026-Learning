import streamlit as st

from rag import process_pdf, rag


# Cấu hình trang
st.set_page_config(page_title="PDF RAG Chatbot", layout="wide")
st.title("PDF RAG Chatbot")

# Giữ dữ liệu PDF và lịch sử chat khi Streamlit chạy lại trang
for k, v in {"collection": None, "pdf_name": "", "chat_history": []}.items():
    st.session_state.setdefault(k, v)

# Sidebar: upload PDF và nút điều khiển
with st.sidebar:
    st.subheader("Upload tài liệu")
    f = st.file_uploader("Chọn file PDF", type="pdf")

    if f and st.button("Xử lý PDF", use_container_width=True):
        with st.spinner("Đang xử lý..."):
            try:
                st.session_state.collection, n = process_pdf(f)
                st.session_state.pdf_name = f.name
                st.session_state.chat_history = []
                st.success(f"Đã index {n} chunks")
            except ValueError as e:
                st.error(str(e))

    if st.session_state.pdf_name:
        st.info(st.session_state.pdf_name)
    else:
        st.info("Chưa có tài liệu")

    if st.button("Xóa lịch sử chat", use_container_width=True):
        st.session_state.chat_history = []


# Hiển thị lịch sử chat
for m in st.session_state.chat_history:
    with st.chat_message(m["role"]):
        st.write(m["content"])


# Ô nhập câu hỏi
if st.session_state.collection is None:
    st.info("Upload và xử lý PDF trước khi chat.")
    st.chat_input("Nhập câu hỏi...", disabled=True)
else:
    q = st.chat_input("Nhập câu hỏi của bạn...")
    if q:
        st.session_state.chat_history.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.write(q)

        with st.chat_message("assistant"):
            with st.spinner("Đang suy nghĩ..."):
                ans = rag(q, st.session_state.collection)
            st.write(ans)

        st.session_state.chat_history.append({"role": "assistant", "content": ans})
