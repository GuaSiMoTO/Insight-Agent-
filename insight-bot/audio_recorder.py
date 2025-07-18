import sounddevice as sd
import numpy as np
import tempfile
import wave
import threading
import time

class AudioRecorder:
    def __init__(self, sample_rate=16000, channels=1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        self.frames = []
        self.recording = False
        self.thread = None

    def _callback(self, indata, frames, time_info, status):
        if self.recording:
            self.frames.append(indata.copy())

    def start(self):
        self.recording = True
        self.thread = threading.Thread(target=self._record)
        self.thread.start()

    def _record(self):
        with sd.InputStream(samplerate=self.sample_rate, channels=self.channels, callback=self._callback):
            while self.recording:
                sd.sleep(100)
        self.save()

    def stop(self):
        self.recording = False
        if self.thread:
            self.thread.join()

    def save(self):
        audio = np.concatenate(self.frames, axis=0)
        with wave.open(self.temp_wav.name, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(self.sample_rate)
            wf.writeframes((audio * 32767).astype(np.int16).tobytes())
        return self.temp_wav.name

    def get_temp_wav_path(self):
        return self.temp_wav.name
