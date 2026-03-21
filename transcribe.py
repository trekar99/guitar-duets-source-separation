import os
import mido
from basic_pitch.inference import predict
import librosa

INPUT_FOLDER = "GuitarDuets/Real/test"
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
    return float(tempo)

def process_file(file_path):
    print(f"\nProcessing {file_path} ...")
    BPM = estimate_bpm(file_path)
    print(f"Estimated BPM: {BPM:.2f}")

    filename_lower = os.path.basename(file_path).lower()
    instrument_name = "guitar1" if "guitar1" in filename_lower else "guitar2" if "guitar2" in filename_lower else "unknown"

    model_output, midi_data, note_events = predict(file_path)
    print(f"Detected {len(note_events)} notes")

    mid = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    # Instrument name
    track.append(mido.MetaMessage('instrument_name', name=instrument_name, time=0))
    # Tempo
    tempo_us_per_beat = int(60_000_000 / BPM)
    track.append(mido.MetaMessage('set_tempo', tempo=tempo_us_per_beat, time=0))

    last_tick = 0  # absolute ticks of previous message
    for note in note_events:
        start, end, pitch, amplitude = note[:4]
        velocity = max(30, min(127, int(amplitude * 127)))

        start_q = quantize_time(start, BPM)
        end_q = quantize_time(end, BPM)
        if end_q <= start_q:
            continue

        duration_sec = end_q - start_q
        duration_beats = duration_sec * BPM / 60
        if duration_beats < MIN_INTERVAL_BEATS:
            continue

        label, quantized_beats = quantize_duration(duration_beats)
        if quantized_beats < MIN_INTERVAL_BEATS:
            continue

        end_q = start_q + quantized_beats * 60 / BPM

        start_tick = int(start_q * BPM / 60 * TICKS_PER_BEAT)
        end_tick = int(end_q * BPM / 60 * TICKS_PER_BEAT)

        # Ensure delta times are non-negative
        delta_on = max(start_tick - last_tick, 0)
        track.append(mido.Message('note_on', note=pitch, velocity=velocity, time=delta_on))

        delta_off = max(end_tick - start_tick, 0)
        track.append(mido.Message('note_off', note=pitch, velocity=0, time=delta_off))

        last_tick = end_tick  # update absolute tick

    output_file = os.path.splitext(file_path)[0] + "_quantized.mid"
    mid.save(output_file)
    print(f"Saved MIDI to {output_file}")

def main():
    for root, dirs, files in os.walk(INPUT_FOLDER):
        for f in files:
            if f.lower().endswith(".wav") and f.lower().startswith("guitar"):
                process_file(os.path.join(root, f))
    print("\nAll files processed ✅")

if __name__ == "__main__":
    main()