import streamlit as st

from src import RAGService


st.set_page_config(page_title="PDF RAG Chatbot", layout="wide")


@st.cache_resource
def get_rag_service():
    return RAGService()


def show_citations(citations):
    if not citations:
        return

    with st.expander("Sources"):
        for idx, citation in enumerate(citations, start=1):
            st.markdown(f"**[{idx}] {citation.file_name}, page {citation.page}**")
            st.caption(citation.snippet)


def format_latex(text):
    """Convert common LaTeX delimiters for Streamlit Markdown."""
    return (
        text.replace("\\(", "$")
        .replace("\\)", "$")
        .replace("\\[", "$$")
        .replace("\\]", "$$")
    )


def show_indexing_report(report):
    """Show a short indexing summary in the sidebar."""
    st.caption(
        f"{report.source_count} PDFs · {report.page_count} pages · "
        f"{report.chunk_count} chunks"
    )
    st.caption(f"Empty pages: {report.empty_page_count}")


st.title("PDF RAG Chatbot")

rag_service = get_rag_service()
st.session_state.setdefault("notebook", None)
st.session_state.setdefault("chat_history", [])

with st.sidebar:
    st.subheader("Notebook sources")
    files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True,
    )

    if files and st.button("Process notebook", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_indexing_progress(progress, message):
            percent = int(progress * 100)
            progress_bar.progress(min(max(progress, 0), 1))
            status_text.caption(f"Indexing notebook... {percent}% · {message}")

        try:
            update_indexing_progress(0, "Reading and chunking PDFs")
            st.session_state.notebook = rag_service.process_pdfs(
                files,
                progress_callback=update_indexing_progress,
            )
            st.session_state.chat_history = []
            notebook = st.session_state.notebook
            st.success(
                f"Indexed {notebook.report.chunk_count} chunks from "
                f"{notebook.report.source_count} PDFs"
            )
        except ValueError as e:
            st.error(str(e))

    notebook = st.session_state.notebook
    if notebook:
        st.info("Notebook indexed")
        show_indexing_report(notebook.report)
        for source_name in notebook.source_names:
            st.caption(source_name)
    else:
        st.info("No notebook indexed yet")

    if st.button("Clear chat history", use_container_width=True):
        st.session_state.chat_history = []


for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            st.markdown(format_latex(message["content"]))
            show_citations(message.get("citations", []))
        else:
            st.write(message["content"])


if st.session_state.notebook is None:
    st.info("Upload and process one or more PDFs before chatting.")
    st.chat_input("Ask a question...", disabled=True)
else:
    question = st.chat_input("Ask a question from your sources...")
    if question:
        st.session_state.chat_history.append(
            {"role": "user", "content": question}
        )
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving sources and answering..."):
                response = rag_service.answer_question(question, st.session_state.notebook)
            st.markdown(format_latex(response.answer))
            show_citations(response.citations)

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": response.answer,
                "citations": response.citations,
            }
        )
