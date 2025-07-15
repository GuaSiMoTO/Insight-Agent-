"""RAG chain wrapper."""
from __future__ import annotations
from typing import List

from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from config import GEMINI_MODEL, GOOGLE_API_KEY

_llm = GoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=GOOGLE_API_KEY)
_parser = StrOutputParser()

DEFAULT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", (
            "You are a helpful sales‑assistant AI specialized in Sales. Your role is to help Sales departments to do smart decisiones based on the theory of {context} Use the provided context to answer "
            "the user's question succinctly but completely. If there is not enough information in the context, you can answer with a helpful selling advice.")),
        ("human", "{question}\n\nContext:\n{context}"),
    ]
)


def build_rag_chain(retriever) -> RunnablePassthrough:
    """Returns a LCEL chain: {question} -> RAG answer."""
    return (
        {
            "question": RunnablePassthrough(),
            "context": retriever | (lambda docs: "\n---\n".join(d.page_content for d in docs)),
        }
        | DEFAULT_PROMPT
        | _llm  
        | _parser
    )