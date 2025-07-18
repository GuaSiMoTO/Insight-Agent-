import os

import torch
from transformers import pipeline
from pyannote.audio import Pipeline
# import librosa
# import soundfile as sf
# import numpy as np
import re
import tempfile # Para manejar archivos temporales de forma segura
import glob
import ffmpeg
import whisper

def process_youtube_audio(directory):
    """
    Usa el archivo .m4a más reciente de un directorio, lo convierte a .wav usando ffmpeg-python,
    lo transcribe con Whisper y lo diariza con Pyannote.audio.
    # El resampleo a 16kHz se realiza con ffmpeg, por lo que no es necesario usar librosa/soundfile.
    """
    audio_dir = directory
    m4a_files = glob.glob(os.path.join(audio_dir, '*.mp3'))
    if not m4a_files:
        raise FileNotFoundError(f'No se encontraron archivos .m4a en {audio_dir}')
    latest_m4a = max(m4a_files, key=os.path.getctime)
    print(f"\n--- Usando archivo de audio más reciente: {latest_m4a} ---")
    temp_audio_path = latest_m4a

    try:
        # --- 2. Convertir a WAV usando ffmpeg-python ---
        wav_path = temp_audio_path.replace('.mp3', '.wav')
        (
            ffmpeg.input(temp_audio_path)
            .output(wav_path, acodec='pcm_s16le', ac=1, ar='16k')
            .overwrite_output()
            .run(quiet=True)
        )
        temp_audio_path = wav_path
        print(f"\n--- Audio convertido a WAV y 16kHz: {temp_audio_path} ---")

        # --- 3. Transcribir con Whisper usando transformers (comentado) ---
        # print("\n--- Realizando transcripción con Whisper... ---")
        # transcriber_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-base")
        # whisper_result = transcriber_pipeline(temp_audio_path, return_timestamps=True)
        # full_transcription = whisper_result["text"]
        # print("\n--- Resultado de Transcripción de Whisper ---")
        # print(full_transcription)
        # print("------------------------------------------")

        # --- 3. Transcribir con Whisper (librería oficial) ---
        print("\n--- Realizando transcripción con Whisper (librería oficial)... ---")
        model = whisper.load_model("small")  # Puedes cambiar a "small", "medium", "large" si lo deseas
        whisper_result = model.transcribe(temp_audio_path, fp16=False, language='es')  # fp16=False si no tienes GPU
        full_transcription = whisper_result["text"]
        print("\n--- Resultado de Transcripción de Whisper ---")
        print(full_transcription)
        with open("transcribe/transcription.txt", "w", encoding="utf-8") as f:
            f.write(full_transcription)
        print("\n--- Transcripción guardada en 'transcribe/zoom_audio/transcription.txt' ---")
        print("------------------------------------------")

        # # --- 4. Diarizar con Pyannote.audio ---
        # print("\n--- Realizando diarización con Pyannote.audio (requiere autenticación HF)... ---")
        # diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=True)
        # diarization = diarization_pipeline(temp_audio_path)
        # print("\n--- Resultado de Diarización de Pyannote.audio ---")
        # for segment, _, speaker in diarization.itertracks(yield_label=True):
        #     start_min, start_sec = divmod(segment.start, 60)
        #     end_min, end_sec = divmod(segment.end, 60)
        #     print(f"[{int(start_min):02d}:{int(start_sec):02d} - {int(end_min):02d}:{int(end_sec):02d}] {speaker}")
        # print("------------------------------------------")

    except FileNotFoundError as e:
        print(f"\n❌ Error de archivo no encontrado: {e}")
    except Exception as e:
        print(f"\n❌ Ocurrió un error general: {e}")
    finally:
        # --- Limpiar archivo temporal WAV generado ---
        if temp_audio_path.endswith('.wav') and os.path.exists(temp_audio_path):
    
    # --- Configuración ---

# --- URL del Video de YouTube ---

directory = "transcribe/zoom_audio"

if __name__ == "__main__":
    process_youtube_audio(directory)

# --- URL del Video de YouTube ---







