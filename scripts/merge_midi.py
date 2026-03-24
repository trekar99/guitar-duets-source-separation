import os
import mido

INPUT_FOLDER = "GuitarDuets/Real/test"
TICKS_PER_BEAT = 480
DEFAULT_BPM = 120 

def merge_to_multitrack(folder_path):
    midi_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(".mid") and f.lower() != "notes.mid"
    ]
    
    if len(midi_files) < 2:
        return

    merged_mid = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    
    # Track 0: Global Meta
    meta_track = mido.MidiTrack()
    merged_mid.tracks.append(meta_track)
    meta_track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(DEFAULT_BPM), time=0))
    meta_track.append(mido.MetaMessage('end_of_track', time=0))

    print(f"Finalizing labels for {folder_path}...")

    for mf_path in midi_files:
        mid = mido.MidiFile(mf_path)
        new_track = mido.MidiTrack()
        merged_mid.tracks.append(new_track)
        
        # Determine strict name
        label = "guitar1" if "guitar1" in mf_path.lower() else "guitar2"
        
        # 1. SET BOTH TRACK NAME AND INSTRUMENT NAME
        new_track.append(mido.MetaMessage('track_name', name=label, time=0))
        new_track.append(mido.MetaMessage('instrument_name', name=label, time=0))

        # 2. ALIGNMENT LOGIC
        source_bpm = 120.0
        for track in mid.tracks:
            for msg in track:
                if msg.type == 'set_tempo':
                    source_bpm = mido.tempo2bpm(msg.tempo)
                    break

        events = []
        for track in mid.tracks:
            abs_tick_src = 0
            for msg in track:
                abs_tick_src += msg.time
                if not msg.is_meta and msg.type in ['note_on', 'note_off']:
                    # Convert to absolute seconds then to target ticks
                    seconds = abs_tick_src * (60.0 / (source_bpm * mid.ticks_per_beat))
                    target_abs_tick = int(seconds * (DEFAULT_BPM * TICKS_PER_BEAT / 60.0))
                    events.append(msg.copy(time=target_abs_tick))

        # Sort and Delta-fy
        events.sort(key=lambda x: x.time)
        last_tick = 0
        for msg in events:
            delta = max(0, msg.time - last_tick)
            last_tick = msg.time
            new_track.append(msg.copy(time=delta))

        new_track.append(mido.MetaMessage('end_of_track', time=0))

    output_file = os.path.join(folder_path, "notes.mid")
    merged_mid.save(output_file)
    print(f"   -> Labelled and Merged: {output_file}")

def main():
    for root, dirs, files in os.walk(INPUT_FOLDER):
        midis = [f for f in files if f.lower().endswith(".mid") and f != "notes.mid"]
        if len(midis) >= 2:
            merge_to_multitrack(root)
    print("\nAll folders processed successfully. ✅")

if __name__ == "__main__":
    main()