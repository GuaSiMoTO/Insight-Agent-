from config import Config
from zoom_bot import ZoomBot
from audio_recorder import AudioRecorder
from speaker_diarization import SpeakerDiarization
import os
import time

def main():
    config = Config('config.yaml')
    zoom_url = config.zoom_meeting_url
    output_dir = config.output_dir
    sample_rate = config.sample_rate
    channels = config.channels

    # Start recording
    recorder = AudioRecorder(sample_rate=sample_rate, channels=channels)
    recorder.start()
    print("Recording started...")

    # Join Zoom meeting
    bot = ZoomBot(zoom_url)
    bot.join_meeting()
    print("Joined Zoom meeting.")

    # Wait for the meeting to end (browser window closes)
    try:
        while bot.driver and bot.driver.service.is_connectable():
            time.sleep(5)
    except Exception:
        pass
    print("Meeting ended or browser closed.")

    # Stop recording
    recorder.stop()
    print("Recording finished.")
    bot.close()

    wav_path = recorder.get_temp_wav_path()
    print(f"Audio saved to {wav_path}")

    # Speaker diarization
    diarizer = SpeakerDiarization()
    segments = diarizer.diarize(wav_path)
    print("Speaker segments:", segments)
    diarizer.split_audio_by_speaker(wav_path, output_dir)
    print(f"Speaker audio files saved to {output_dir}")

if __name__ == "__main__":
    main()
