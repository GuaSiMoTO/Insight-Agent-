"""PDF helpers."""
from pathlib import Path
from typing import List
from pypdf import PdfReader
from langchain.schema import Document
from langchain.document_loaders import PyPDFLoader

def pdf_to_text(path: Path) -> str:
    """Extracts *all* text from a single PDF file."""
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)




def load_pdfs(pdf_paths: List[Path]) -> List[Document]:
    all_docs = []
    for path in pdf_paths:
        loader = PyPDFLoader(str(path))
        docs = loader.load_and_split()
        for d in docs:
            d.metadata["source"] = path.name
            d.metadata["file_path"] = str(path)
            d.metadata["page_number"] = d.metadata.get("page", 0)  # Default to 0 if not present
            d.metadata['doc_type'] = 'theory' # Add doc_type metadata   
        all_docs.extend(docs)
    return all_docs