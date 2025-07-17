import requests
from typing import Tuple
from API.app.config import N8N_WEBHOOK_URL  # Importa la URL del webhook de n8n desde tu configuración

def enviar_texto_a_n8n(contenido: str) -> Tuple[int, str]:
    """
    Envía el contenido de texto a n8n a través de un webhook.
    
    Args:
        contenido (str): El contenido de texto a enviar.
    
    Returns:
        Tuple[int, str]: El código de estado HTTP y la respuesta del servidor.
    """
    from API.app.config import N8N_WEBHOOK_URL

    payload = {
        "contenido": contenido
    }

    try:
        response = requests.post(N8N_WEBHOOK_URL, json=payload)
        return response.status_code, response.text
    except Exception as e:
        return 500, f"Error al enviar a n8n: {str(e)}"