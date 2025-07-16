import os
import subprocess
import torch
from transformers import pipeline
from pyannote.audio import Pipeline
import librosa
import soundfile as sf
import numpy as np
import re
import tempfile # Para manejar archivos temporales de forma segura

def process_youtube_audio(youtube_url):
    """
    Descarga el audio de un video de YouTube, lo transcribe con Whisper
    y lo diariza con Pyannote.audio, imprimiendo ambos resultados.
    """
    
    # --- Configuración ---
    
    temp_audio_file_obj = None # Usaremos el objeto de archivo temporal
    temp_audio_path = None
    
    try:
        # --- 1. Descargar Audio de YouTube ---
        print("\n--- Descargando audio de YouTube... ---")
        
        # Crear un archivo temporal para guardar el audio
        temp_audio_file_obj = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_audio_path = temp_audio_file_obj.name
        temp_audio_file_obj.close() # Es importante cerrarlo para que yt-dlp pueda escribir en él

        # Comando para descargar el audio usando yt-dlp
        command = [
            "yt-dlp",
            "-x", # Extraer audio
            "--audio-format", "wav", # Formato de audio WAV
            "--no-mtime", # Evitar problemas de marcas de tiempo
            "--force-overwrites", # Forzar la sobrescritura de archivos existentes
            "-o", temp_audio_path, # Ruta de salida del archivo
            youtube_url
        ]
        
        # Ejecutamos el comando y capturamos la salida para depuración
        result = subprocess.run(command, capture_output=True, text=True, check=False)

        if result.returncode != 0:
            print(f"❌ yt-dlp falló con código de salida {result.returncode}")
            print("--- Salida estándar de yt-dlp ---")
            print(result.stdout)
            print("--- Salida de error de yt-dlp ---")
            print(result.stderr)
            raise Exception(f"Error al descargar o procesar el audio de YouTube. Revisa la salida de yt-dlp anterior.")
        
        print(f"✅ Audio descargado en: {temp_audio_path}")

        # --- 2. Preprocesar Audio (Resample a 16kHz para modelos ML) ---
        print("\n--- Preprocesando audio (resampleando a 16kHz)... ---")
        y, sr = librosa.load(temp_audio_path, sr=16000)
        sf.write(temp_audio_path, y, sr) # Sobrescribir el archivo con la versión resampleada
        print("✅ Audio preprocesado a 16kHz.")

        # --- 3. Transcribir con Whisper ---
        print("\n--- Realizando transcripción con Whisper... ---")
        # Inicializar el pipeline de Whisper.
        transcriber_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-small")
        # 'return_timestamps=True' es crucial para audios largos.
      
        whisper_result = transcriber_pipeline(temp_audio_path, return_timestamps=True)
        full_transcription = whisper_result["text"]
        
        print("\n--- Resultado de Transcripción de Whisper ---")
        print(full_transcription)
        print("------------------------------------------")

        # --- 4. Diarizar con Pyannote.audio ---
        print("\n--- Realizando diarización con Pyannote.audio (requiere autenticación HF)... ---")
        # use_auth_token=True indica a pyannote que busque el token en la caché de Hugging Face
        diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=True)
        
        # Ejecutar la diarización
        diarization = diarization_pipeline(temp_audio_path)
        
        print("\n--- Resultado de Diarización de Pyannote.audio ---")
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            start_min, start_sec = divmod(segment.start, 60)
            end_min, end_sec = divmod(segment.end, 60)
            print(f"[{int(start_min):02d}:{int(start_sec):02d} - {int(end_min):02d}:{int(end_sec):02d}] {speaker}")
        print("------------------------------------------")

    except FileNotFoundError as e:
        if "yt-dlp" in str(e):
            print("\n❌ ERROR: 'yt-dlp' no se encontró. Asegúrate de que esté instalado y en tu PATH.")
            print("Puedes instalarlo con: pip install yt-dlp")
        else:
            print(f"\n❌ Error de archivo no encontrado: {e}")
    except Exception as e:
        print(f"\n❌ Ocurrió un error general: {e}")
    finally:
        # --- 5. Limpiar Archivo Temporal ---
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
            print(f"\n✅ Archivo temporal eliminado: {temp_audio_path}")

# --- URL del Video de YouTube ---
# Sugerencia: Busca un video de entrevista o de gente hablando para ver bien la diarización.
youtube_video_url = "https://youtu.be/GP3zSH-vjmM?si=1eHeCKok2bUOSDir"


if __name__ == "__main__":
    process_youtube_audio(youtube_video_url)






 