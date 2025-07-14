import streamlit as st
from pathlib import Path
from tempfile import NamedTemporaryFile

from embedder import ingest_pdfs
from rag import build_rag_chain

st.title("Sales AI RAG Agent")

uploaded_files = st.file_uploader(
    "Upload PDF files", type="pdf", accept_multiple_files=True
)

if uploaded_files:
    with st.spinner("Processing PDFs..."):
        temp_paths = []
        for uploaded_file in uploaded_files:
            with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getbuffer())
                temp_paths.append(Path(tmp.name))

        store = ingest_pdfs(temp_paths)
        retriever = store.as_retriever()
        rag_chain = build_rag_chain(retriever)

        st.success("PDFs ingested. Ask your questions below.")

        question = st.text_input("Your question:")

        if question:
            with st.spinner("Getting answer..."):
                answer = rag_chain.invoke(question)
            st.markdown(f"**Answer:** {answer}")

        # Uncomment to delete temp files after use
        # import os
        # for p in temp_paths:
        #     os.unlink(p)
else:
    st.info("Please upload one or more PDF files to get started.")
