"""CLI entry‑point."""
import argparse
from pathlib import Path

from embedder import ingest_pdfs
from rag import build_rag_chain


def main() -> None:
    parser = argparse.ArgumentParser(description="Sales AI RAG agent")
    parser.add_argument("pdfs", nargs="+", type=Path, help="PDF files to ingest")
    args = parser.parse_args()

    # 1. Ingest and embed PDFs
    store = ingest_pdfs(args.pdfs)
    retriever = store.as_retriever()

    # 2. Build RAG chain
    rag_chain = build_rag_chain(retriever)

    # 3. Simple REPL
    print("Ingestion complete ✔️  Ask your questions (type 'exit' to quit)")
    while True:
        question = input("\n🗣️  > ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        answer = rag_chain.invoke(question)
        print(f"\n🤖  {answer}\n")


if __name__ == "__main__":
    main()