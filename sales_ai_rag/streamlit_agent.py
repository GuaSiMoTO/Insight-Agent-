import streamlit as st
from rag import build_rag_chain
from langchain_core.prompts import ChatPromptTemplate
from embedder import ingest_pdfs
from pathlib import Path
from tempfile import NamedTemporaryFile

st.title("Sales AI RAG Agent")
store = ingest_pdfs(temp_paths)
retriever = store.as_retriever()
rag_chain = build_rag_chain(retriever)

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