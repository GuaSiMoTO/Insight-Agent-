import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from API.app.sender import enviar_texto_a_n8n

TRANSCRIPTION_FILE = "transcribe/transcription.txt"

class TranscriptionHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith("transcription.txt"):
            print("📄 Transcripción modificada. Enviando a n8n...")
            try:
                with open(TRANSCRIPTION_FILE, "r", encoding="utf-8") as f:
                    contenido = f.read()
                status, resp = enviar_texto_a_n8n(contenido)
                print(f"✅ Enviado a n8n (status {status})")
            except Exception as e:
                print(f"❌ Error al enviar: {e}")

if __name__ == "__main__":
    event_handler = TranscriptionHandler()
    observer = Observer()
    observer.schedule(event_handler, path="transcribe", recursive=False)
    observer.start()
    print("👁️‍🗨️ Observando archivo de transcripción...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
