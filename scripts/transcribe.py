import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import mido
import librosa
import numpy as np


try:
    from basic_pitch.inference import predict as _bp_predict
    _HAS_BASIC_PITCH = True
except ImportError:
    _HAS_BASIC_PITCH = False
    _bp_predict = None

from pathlib import Path

# --- CONFIG ---
# Ensure this path is 100% correct relative to where you run the command
INPUT_FOLDER = "GuitarDuets/Real/test" 
TICKS_PER_BEAT = 480

def estimate_bpm(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = tempo[0] if isinstance(tempo, (list, np.ndarray)) else tempo
    return float(bpm)

def process_file(file_path):
    if not _HAS_BASIC_PITCH:
        raise RuntimeError(
            "basic-pitch is required for scripts/transcribe.py but is not installed.\n"
            "Install it on Python ≤3.12:  pip install 'basic-pitch[torch]'"
        )
    print(f"-> Found: {os.path.basename(file_path)}")
    bpm = estimate_bpm(file_path)
    
    # Inference (Runs on CPU if GPU fails, but it WILL run)
    _, _, note_events = _bp_predict(file_path)
    
    mid = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(bpm), time=0))

    # Collect and Sort Events
    events = []
# Change this line in your loop:
    for start, end, pitch, amplitude, _ in note_events:  # Added the fifth variable '_'
        if end <= start: 
            continue
            
        velocity = int(amplitude * 127)
        # Ensure pitch is an integer for MIDI
        pitch_int = int(round(pitch)) 
        
        events.append({'type': 'note_on', 'pitch': pitch_int, 'vel': velocity, 'time': start})
        events.append({'type': 'note_off', 'pitch': pitch_int, 'vel': 0, 'time': end})

    events.sort(key=lambda x: x['time'])

    last_tick = 0
    for event in events:
        abs_tick = int(event['time'] * bpm / 60 * TICKS_PER_BEAT)
        delta_tick = max(0, abs_tick - last_tick)
        track.append(mido.Message(event['type'], note=event['pitch'], velocity=event['vel'], time=delta_tick))
        last_tick = abs_tick

    output_file = os.path.splitext(file_path)[0] + ".mid"
    mid.save(output_file)
    print(f"   SUCCESS: Saved to {os.path.basename(output_file)}")

def main():
    path = Path(INPUT_FOLDER)
    if not path.exists():
        print(f"ERROR: The folder '{INPUT_FOLDER}' does not exist!")
        return

    # Use rglob to find wav files recursively
    files_to_process = list(path.rglob("guitar*.wav"))
    
    if not files_to_process:
        print(f"No files starting with 'guitar' and ending in '.wav' found in {INPUT_FOLDER}")
        return

    print(f"Starting transcription of {len(files_to_process)} files...")
    for f in files_to_process:
        process_file(str(f))
    print("\nDone! ✅")

if __name__ == "__main__":
    main()