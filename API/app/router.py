from fastapi import APIRouter, UploadFile, File
from sender import enviar_texto_a_n8n

router = APIRouter()

@router.post("/enviar-txt/")
async def enviar_txt(file: UploadFile = File(...)):
    contenido_bytes = await file.read()
    contenido_texto = contenido_bytes.decode("utf-8")  # Asegúrate de que esté en UTF-8
    status, respuesta = enviar_texto_a_n8n(contenido_texto)
    return {"status": status, "respuesta": respuesta}