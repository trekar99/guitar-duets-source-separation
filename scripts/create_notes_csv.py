from __future__ import annotations

import argparse
from pathlib import Path

import mido
import pandas as pd

from _bootstrap import bootstrap

bootstrap()


def get_tempo(mid):
    for track in mid.tracks:
        for msg in track:
            if msg.type == "set_tempo":
                return msg.tempo
    raise ValueError("No tempo event found in MIDI file.")


def midi_to_csv(midi_file: Path, csv_output: Path, sample_rate: int = 44100) -> None:
    mid = mido.MidiFile(str(midi_file))
    tempo = get_tempo(mid)

    note_events = []
    current_notes = {}

    for track in mid.tracks:
        absolute_time = 0
        instrument = None
        for msg in track:
            absolute_time += msg.time
            if msg.type == "instrument_name":
                instrument = 1 if msg.name == "guitar1" else 2
            if msg.type == "note_on":
                note = msg.note
                if msg.velocity == 0:
                    if note in current_notes:
                        start_time = current_notes[note][0]
                        inst = current_notes[note][1]
                        note_events.append((start_time, absolute_time, inst, note))
                        del current_notes[note]
                else:
                    if note in current_notes:
                        start_time = current_notes[note][0]
                        prev_instrument = current_notes[note][1]
                        note_events.append((start_time, absolute_time, prev_instrument, note))
                    current_notes[note] = (absolute_time, instrument)

    note_events_samples = [
        (
            int(mido.tick2second(start, mid.ticks_per_beat, tempo) * sample_rate),
            int(mido.tick2second(end, mid.ticks_per_beat, tempo) * sample_rate),
            instrument,
            note,
        )
        for start, end, instrument, note in note_events
    ]

    dataframe = pd.DataFrame(
        note_events_samples, columns=["start_time", "end_time", "instrument", "note"])
    dataframe = dataframe[dataframe["start_time"] != dataframe["end_time"]]
    dataframe = dataframe.sort_values("start_time")
    dataframe.to_csv(csv_output, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate notes.csv from notes.mid for each track.")
    parser.add_argument("--dataset-root", required=True, help="Root dataset directory to scan.")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    for midi_path in dataset_root.rglob("notes.mid"):
        csv_output = midi_path.with_name("notes.csv")
        midi_to_csv(midi_path, csv_output)
        print(f"Wrote {csv_output}")


if __name__ == "__main__":
    main()
