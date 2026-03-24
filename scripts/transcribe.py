import os

# --- SILENCE THE NOISE (Add this at the very top) ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import mido
import librosa
import numpy as np
from pathlib import Path


try:
    from basic_pitch.inference import predict as _bp_predict
    _HAS_BASIC_PITCH = True
except ImportError:
    _HAS_BASIC_PITCH = False
    _bp_predict = None

INPUT_FOLDER = "dataset/Real/test"
TICKS_PER_BEAT = 480  # standard PPQ

# --- rhythm intervals / quantization ---
INTERVALS = {
    "Triplet Sixty Fourth": (0.037125, 0.041875),
    "Sixty Fourth": (0.05625, 0.0625),
    "Triplet Thirty Second": (0.07425, 0.09375),
    "Thirty Second": (0.1125, 0.135),
    "Triplet Sixteenth": (0.1485, 0.1975),
    "Sixteenth": (0.225, 0.28),
    "Triplet": (0.297, 0.3425),
    "Dotted Sixteenth": (0.3375, 0.4275),
    "Eighth": (0.45, 0.615),
    "Dotted Eighth": (0.675, 0.865),
    "Quarter": (0.9, 1.0525),
    "Tied Quarter-Thirty Second": (1.0125, 1.1775),
    "Tied Quarter-Sixteenth": (1.125, 1.365),
    "Dotted Quarter": (1.35, 1.74),
    "Half": (1.8, 2.49),
    "Dotted Half": (2.7, 3.49),
    "Whole": (3.6, 4.4),
}

INTERVAL_CENTERS = {name: (low + high) / 2 for name, (low, high) in INTERVALS.items()}
MIN_INTERVAL_BEATS = min(low for low, high in INTERVALS.values())

def quantize_duration(duration_beats):
    best_label = None
    best_dist = float("inf")
    for label, center in INTERVAL_CENTERS.items():
        low, high = INTERVALS[label]
        low *= 0.95
        high *= 1.05
        if low <= duration_beats <= high:
            return label, center
        dist = abs(duration_beats - center)
        if dist < best_dist:
            best_dist = dist
            best_label = label
    return best_label, INTERVAL_CENTERS[best_label]

def quantize_time(time_sec, bpm, grid=0.25):
    beats = time_sec * bpm / 60
    quantized_beats = round(beats / grid) * grid
    return quantized_beats * 60 / bpm

def estimate_bpm(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = tempo[0] if isinstance(tempo, (list, np.ndarray)) else tempo
    return float(bpm)

def process_file(file_path):
    print(f"-> Found: {os.path.basename(file_path)}")
    bpm = estimate_bpm(file_path)
    
    # Inference (Runs on CPU if GPU fails, but it WILL run)    
    if not _HAS_BASIC_PITCH:
        raise RuntimeError(
            "basic-pitch is required for scripts/transcribe.py but is not installed.\n"
            "Install it on Python ≤3.12:  pip install 'basic-pitch[torch]'"
        )

    print(f"\nProcessing {file_path} ...")
    BPM = estimate_bpm(file_path)
    print(f"Estimated BPM: {BPM:.2f}")

    filename_lower = os.path.basename(file_path).lower()
    instrument_name = "guitar1" if "guitar1" in filename_lower else "guitar2" if "guitar2" in filename_lower else "unknown"

    _, _, note_events = _bp_predict(file_path)
    print(f"Detected {len(note_events)} notes")

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