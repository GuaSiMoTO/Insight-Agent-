"""PDF helpers."""
from pathlib import Path
from typing import List
from pypdf import PdfReader

def pdf_to_text(path: Path) -> str:
    """Extracts *all* text from a single PDF file."""
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def load_pdfs(paths: List[Path]) -> str:
    """Concatenate text from multiple PDFs into a single string."""
    texts = [pdf_to_text(p) for p in paths]
    return "\n".join(texts)