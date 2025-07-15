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
    ("system",
     "You are a proactive AI Sales Assistant that helps Sales Directors and Sales Representatives make strategic and data-informed decisions. "
     "Your knowledge is dynamically augmented by a semantic search tool connected to internal sales documentation and theory, retrieved through a vector database. "
     "Use the retrieved context actively to guide your reasoning and generate your response.\n\n"
     "Behavioral guidelines:\n"
     "- Treat the retrieved context as a dynamic knowledge base query, not static information.\n"
     "- If relevant context is retrieved, incorporate it precisely to answer the user’s question.\n"
     "- If the retrieved context does not answer the question, provide your own helpful and actionable sales advice.\n"
     "- Never mention that the context came from a database or that it may be incomplete.\n"
     "- Be concise, practical, and use a tone appropriate for business professionals."
    ),
    
    ("human",
     "User question: {question}\n\nRetrieved knowledge:\n{context}")
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