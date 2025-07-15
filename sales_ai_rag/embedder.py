"""Chunk, embed, and persist to Qdrant."""
from __future__ import annotations
from typing import List
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore

from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBED_MODEL,
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    GOOGLE_API_KEY
)
from pdf_utils import load_pdfs

# --- Splitter & Embeddings ---------------------------------------------------
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", " ", ""]
)
_embedding = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL, google_api_key=GOOGLE_API_KEY)


def ingest_pdfs(pdf_paths: List[Path]) -> QdrantVectorStore:
    """Read PDFs → split → embed → upsert into Qdrant. Returns the store."""
    raw_text: List[Document] = load_pdfs(pdf_paths)
    docs: List[Document] = _splitter.split_documents(raw_text)

    # ⚠️ If you want metadata (page numbers, file names) add them here
    # for d in docs: d.metadata = {...}
    vector_store = QdrantVectorStore.from_documents(
        documents=docs,
        embedding=_embedding,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=QDRANT_COLLECTION,
    )
    return vector_store