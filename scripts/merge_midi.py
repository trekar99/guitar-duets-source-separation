import os
import mido

# =========================
# CONFIG
# =========================
INPUT_FOLDER = "dataset/Real/test"
TICKS_PER_BEAT = 480  # standard MIDI resolution

# =========================
# FUNCTION TO MERGE MIDIS IN A FOLDER
# =========================
def merge_midis_in_folder(folder_path):
    midi_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith("_quantized.mid")
    ]
    if not midi_files:
        return  # nothing to merge

    merged = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    merged_track = mido.MidiTrack()
    merged.tracks.append(merged_track)

    print(f"Merging {len(midi_files)} files in {folder_path}:")

    for mf in midi_files:
        mid = mido.MidiFile(mf)
        for track in mid.tracks:
            for msg in track:
                # Copy all messages to merged track
                merged_track.append(msg)
                # Print instrument names
                if msg.type == "instrument_name":
                    print(f" - Found instrument: {msg.name} in {mf}")

    output_file = os.path.join(folder_path, "notes.mid")
    merged.save(output_file)
    print(f"-> Merged file written: {output_file}")

# =========================
# MAIN
# =========================
def main():
    for root, dirs, files in os.walk(INPUT_FOLDER):
        if any(f.lower().endswith("_quantized.mid") for f in files):
            merge_midis_in_folder(root)
    print("\nAll folders processed ✅")

if __name__ == "__main__":
    main()