import os
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate

# --- Configuración inicial ---
llm = GoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key='AIzaSyB1379yvRoIEpbZe7FQKrt-lMLxHiQH_X8'
)

# --- 1. Definir la carpeta donde están tus PDFs ---
# Asegúrate de que esta carpeta exista y contenga tus archivos PDF.
pdf_folder = "PDFs" # Nombre de la carpeta que contendrá tus PDFs
# Por ejemplo, si tu script está en /mi_proyecto/script.py
# y tus PDFs están en /mi_proyecto/mis_pdfs/documento1.pdf, documento2.pdf

print(f"--- Buscando PDFs en la carpeta: {pdf_folder} ---")

# Obtener una lista de todos los archivos PDF en la carpeta
pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]

if not pdf_files:
    print(f"No se encontraron archivos PDF en la carpeta '{pdf_folder}'. Asegúrate de que la carpeta exista y contenga PDFs.")
    exit()

print(f"Archivos PDF encontrados: {pdf_files}")

# --- 2. Cargar el contenido completo de todos los PDFs ---

print("\n--- Cargando contenido de los PDFs ---")
full_combined_pdf_content = ""
for pdf_file in pdf_files:
    pdf_path = os.path.join(pdf_folder, pdf_file)
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load() # Carga el contenido del PDF como objetos Document
        for doc in documents:
            full_combined_pdf_content += doc.page_content + "\n" # Unir el contenido de todas las páginas
        print(f"Contenido de '{pdf_file}' cargado exitosamente.")
    except Exception as e:
        print(f"Error al cargar o procesar '{pdf_file}': {e}")

if not full_combined_pdf_content:
    print("No se pudo cargar contenido de ningún PDF.")
    exit()

# Opcional: Para verificar el contenido y la longitud
# print(f"\nPrimeros 500 caracteres del contenido combinado:\n{full_combined_pdf_content[:500]}...")
# print(f"\nLongitud total del contenido combinado de los PDFs: {len(full_combined_pdf_content)} caracteres")


# --- 3. Preparar el Prompt con el Contenido del PDF ---

print("\n--- Preparando el prompt para el LLM ---")
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "Eres un asistente útil. Responde a la pregunta basándote únicamente en el siguiente contenido. Si la respuesta no está en el contenido, indica que no tienes esa información."),
    ("user", "Contenido:\n{context}\n\nPregunta: {question}")
])

# Crea la cadena simple: prompt -> llm -> parser
chain = prompt_template | llm | StrOutputParser()

# --- 4. Preguntar al LLM usando el contenido del PDF como contexto ---

print("\n--- Preguntando al LLM usando el contenido de los PDFs como contexto ---")
user_question = "¿Puedes resumirme la información principal de estos documentos?"
# user_question = "¿Qué se dice sobre [algún tema específico] en los PDFs?" # Adapta tu pregunta

try:
    # Invocar la cadena, pasando el contenido completo combinado de los PDFs como 'context'
    response = chain.invoke({"context": full_combined_pdf_content, "question": user_question})
    print(f"Pregunta del usuario: {user_question}")
    print(f"Respuesta del LLM:\n{response}")
except Exception as e:
    print(f"Error al obtener respuesta del LLM: {e}")