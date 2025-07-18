from pyannote.audio import Pipeline
import tempfile
import os

class SpeakerDiarization:
    def __init__(self, model_name="pyannote/speaker-diarization@2.1"):
        self.pipeline = Pipeline.from_pretrained(model_name)

    def diarize(self, wav_path):
        diarization = self.pipeline(wav_path)
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                'start': turn.start,
                'end': turn.end,
                'speaker': speaker
            })
        return segments

    def split_audio_by_speaker(self, wav_path, output_dir):
        diarization = self.pipeline(wav_path)
        speaker_files = {}
        with tempfile.TemporaryDirectory() as tmpdir:
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                out_path = os.path.join(output_dir, f"speaker_{speaker}.wav")
                # Use ffmpeg to extract segments (requires ffmpeg installed)
                os.system(f"ffmpeg -y -i {wav_path} -ss {turn.start} -to {turn.end} -c copy {out_path}")
                if speaker not in speaker_files:
                    speaker_files[speaker] = []
                speaker_files[speaker].append(out_path)
        return speaker_files
