import os
import streamlit as st # Importamos Streamlit
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate

# --- Configuración inicial de Streamlit ---
st.set_page_config(page_title="Chat con tus PDFs (Langchain + Gemini)", layout="wide")
st.title("📚 Chat con tus PDFs")
st.write("Sube tus documentos PDF a la carpeta 'PDFs' y pregúntame lo que quieras sobre ellos.")

# --- Configuración inicial del LLM ---
# ¡IMPORTANTE!: No compartas tu API key en código público.
# Considera usar st.secrets para producción o variables de entorno.
llm = GoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key='AIzaSyC8DBcCUTxAWsrVfyqk1s167x8Fi0kjS5Y'
)

# --- 1. Definir la carpeta donde están tus PDFs ---
pdf_folder = "PDFs"

# Inicializar estado de la sesión para el cLimpiontenido del PDF y el historial del chat
if 'full_combined_pdf_content' not in st.session_state:
    st.session_state.full_combined_pdf_content = ""
if 'transcript' not in st.session_state:
    st.session_state.transcript = ""
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Función para cargar PDFs (la ponemos en caché para que no se recargue cada vez)
@st.cache_resource
def load_pdfs(folder):
    """Carga y combina el contenido de todos los PDFs en la carpeta especificada."""
    st.info(f"--- Buscando PDFs en la carpeta: {folder} ---")
    pdf_files = [f for f in os.listdir(folder) if f.lower().endswith('.pdf')]

    if not pdf_files:
        st.error(f"No se encontraron archivos PDF en la carpeta '{folder}'. Asegúrate de que la carpeta exista y contenga PDFs.")
        return ""

    st.success(f"Archivos PDF encontrados: {pdf_files}")
    st.info("\n--- Cargando contenido de los PDFs (esto puede tardar) ---")
    combined_content = ""
    progress_bar = st.progress(0)
    for i, pdf_file in enumerate(pdf_files):
        pdf_path = os.path.join(folder, pdf_file)
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            for doc in documents:
                combined_content += doc.page_content + "\n"
            st.write(f"✅ Contenido de '{pdf_file}' cargado.")
        except Exception as e:
            st.error(f"❌ Error al cargar o procesar '{pdf_file}': {e}")
        progress_bar.progress((i + 1) / len(pdf_files))

    if not combined_content:
        st.error("No se pudo cargar contenido de ningún PDF.")
        return ""
    st.success("✅ Contenido de todos los PDFs cargado exitosamente.")
    return combined_content

@st.cache_resource
def load_transcript():
    with open("transcribe/transcription.txt", "r", encoding="utf-8") as transcript_file:
        transcript = transcript_file.read()
        return transcript
    

# Cargar el contenido combinado de los PDFs al inicio
# Cargar PDFs al inicio de la aplicación si no están cargados
if not st.session_state.full_combined_pdf_content:
    st.session_state.full_combined_pdf_content = load_pdfs(pdf_folder)
    if not st.session_state.full_combined_pdf_content:
        st.stop() # Detener la ejecución si no hay contenido cargado

if not st.session_state.transcript:
    st.session_state.transcript = load_transcript()
    if not st.session_state.transcript:
        st.stop() # Detener la ejecución si no hay transcripción cargada

# --- Preparar el Prompt con el Contenido del PDF ---
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """Eres un asistente útil. Responde a la pregunta basándote únicamente en el siguiente contenido.

Actúa como Insight Agent, un observador inteligente especializado en ventas consultivas avanzadas. Acabas de escuchar una reunión interna entre el director comercial y su equipo, donde se han discutido oportunidades de venta.

Tu tarea es analizar la conversación usando las metodologías Insight Selling, Challenger Sale, y marcos de inteligencia competitiva. Usa como referencia la teoría aportada sobre Insight Selling en el siguiente contexto: {context}

Devuelve el siguiente análisis estructurado cuando se te pregunte de insights o estrategias de venta de la reunión::

---

🔥 Resumen Provocador
- ¿Qué parte de la narrativa desafió al cliente o provocó reflexión?
- ¿Se detectó algún insight de negocio o interacción relevante?
- ¿Qué oportunidad latente no fue explotada?

---

📈 Resumen de Posibles Insights
Incluye insights mencionados o detectados implícitamente por sector o cliente:

Insights de Negocio:
- Tendencias o riesgos no considerados por el cliente.
- Comparativas con competidores del sector.
- Cambios regulatorios o de comportamiento del consumidor que justifican acción.

Insights de Interacción:
- Preguntas que llevaron a replantear la situación.
- Momentos de reflexión conjunta o conexión emocional.
- Frases que revelan un punto ciego del cliente.

---

🧠 Score de Narrativa
Evalúa del 0 al 5:
- Tensión Constructiva
- Diferenciación
- Conexión Humana
- Colaboración

---

🧭 Prompts para Coaching
1. ¿Qué idea desafiante podrías insertar en la siguiente conversación con el cliente?
2. ¿Dónde puedes invitar al cliente a co-crear la solución contigo?
3. ¿Qué parte de tu narrativa actual podrías reencuadrar en términos de impacto de negocio?

---

Sé preciso, estratégico y usa lenguaje natural basado en la conversación real.
     Si la respuesta no está en el contenido, indica que no tienes esa información. Si la pregunta no está relacionada con el contenido, indica que no puedes responderla con la información proporcionada.\n\nContenido del documento:\n{context}
     
    Tras dar el resumen con la estructura anterior, responde a las preguntas del usuario de forma directa y concisa, basándote en el contenido del PDF y la transcripción de la reunión. Si no tienes información suficiente, indica que no puedes responder.\n"""),
    ("user", "Pregunta: {question}. Conversación: {transcript}" )
])

# Crea la cadena: prompt -> llm -> parser
chain = prompt_template | llm | StrOutputParser()

# --- Interfaz de Chat de Streamlit ---

# Mostrar mensajes previos del chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Campo de entrada para la pregunta del usuario
user_question = st.chat_input("Escribe tu pregunta sobre los PDFs...")

if user_question:
    # Añadir pregunta del usuario al historial del chat
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            try:
                # Invocar la cadena, pasando el contenido combinado de los PDFs como 'context'
                response = chain.invoke({
                    "context": st.session_state.full_combined_pdf_content,
                    "question": user_question,
                    "transcript": st.session_state.transcript
                })
                st.markdown(response)
                # Añadir respuesta del LLM al historial del chat
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error al obtener respuesta del LLM: {e}")
                st.session_state.messages.append({"role": "assistant", "content": f"Ocurrió un error: {e}"})

# Opcional: Mostrar el contenido combinado (útil para depuración)
with st.expander("Ver Contenido Combinado de PDFs (Solo para depuración)"):
    st.text_area("Contenido:", st.session_state.full_combined_pdf_content, height=300)