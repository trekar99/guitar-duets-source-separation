from __future__ import annotations

import argparse
from pathlib import Path

import mido
import pandas as pd

from _bootstrap import bootstrap

bootstrap()


INTERVALS = {
    "Triplet Sixty Fourth": (0.037125000000000005, 0.041875),
    "Sixty Fourth": (0.05625, 0.06250000000000001),
    "Triplet Thirty Second": (0.07425000000000001, 0.09375000000000001),
    "Thirty Second": (0.1125, 0.135),
    "Triplet Sixteenth": (0.14850000000000002, 0.1975),
    "Sixteenth": (0.225, 0.28),
    "Triplet": (0.29700000000000004, 0.3425),
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


def get_note_value(duration, intervals):
    for note_value, (lower, upper) in intervals.items():
        if lower <= duration <= upper:
            return note_value
    return "Unknown"


def get_tempo(mid):
    for track in mid.tracks:
        for msg in track:
            if msg.type == "set_tempo":
                return msg.tempo
    raise ValueError("No tempo event found in MIDI file.")


def midi_to_csv(midi_file: Path, csv_output: Path) -> None:
    sample_rate = 44100
    mid = mido.MidiFile(str(midi_file))
    tempo = get_tempo(mid)
    seconds_per_beat = 60 / mido.tempo2bpm(tempo)

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
                        instrument = current_notes[note][1]
                        note_events.append((start_time, absolute_time, instrument, note))
                        del current_notes[note]
                else:
                    if note in current_notes:
                        start_time = current_notes[note][0]
                        prev_instrument = current_notes[note][1]
                        note_events.append((start_time, absolute_time, prev_instrument, note))
                    current_notes[note] = (absolute_time, instrument)

    note_events_seconds = [
        (
            mido.tick2second(start, mid.ticks_per_beat, tempo),
            mido.tick2second(end, mid.ticks_per_beat, tempo),
            instrument,
            note,
        )
        for start, end, instrument, note in note_events
    ]
    note_events_samples = [
        (int(start * sample_rate), int(end * sample_rate), instrument, note)
        for start, end, instrument, note in note_events_seconds
    ]

    dataframe = pd.DataFrame(note_events_samples, columns=["start_time", "end_time", "instrument", "note"])
    dataframe["start_beat"] = (dataframe["start_time"] / (sample_rate * seconds_per_beat)).round(2)
    dataframe["end_beat"] = ((dataframe["end_time"] - dataframe["start_time"]) / (sample_rate * seconds_per_beat)).round(2)
    dataframe["note_value"] = dataframe["end_beat"].apply(lambda value: get_note_value(value, INTERVALS))
    dataframe = dataframe[dataframe["end_beat"] >= 0]
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
