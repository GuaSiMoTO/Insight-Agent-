import numpy as numpy
import pandas as pd 
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.docstore.document import Document
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate



text_splitter = RecursiveCharacterTextSplitter(chunk_size = 450, chunk_overlap = 0)



documento = Document(page_content=texto)
splits = text_splitter.split_documents([documento])

for index, split in enumerate(splits):
    print(f"Parte {index + 1}:")
    print(split.page_content)
    print("\n---\n")

chunks = []

for split in splits:
    chunks.append(split.page_content)

df = pd.DataFrame(chunks, columns=["texto"])

llm = GoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key='AIzaSyB1379yvRoIEpbZe7FQKrt-lMLxHiQH_X8'
)

response = llm.invoke('Say Hello')


embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Funcion para crear los emebeddings
def create_embeddings(texts):
    return embeddings.embed_query(texts)

# Calculamos los embeddings
df["embeddings"] = df["texto"].apply(create_embeddings)

# Imprimimos resultados
print("Embeddings calculados:")
df.head()

fragments = []
for _, row in df.iterrows():
    fragments.append(Document(page_content=row["texto"]))

doc_store = QdrantVectorStore.from_documents(
    documents=fragments,
    embedding=embeddings,
    url="https://8243ef5f-44fa-46cf-ac01-057b8e587574.europe-west3-0.gcp.cloud.qdrant.io",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.TgnmoSw0DIaIX_hwflTx2cWSPMhPcLrLFJ6tD_RwtDk",
    collection_name="Store"
)

output_parser = StrOutputParser()

prompt = ChatPromptTemplate.from_messages([
    
])

chain = prompt | llm | output_parser