# GuitarDuets Repository

This repo is the code repository of the publication:

Classical Guitar Duet Separation using GuitarDuets -- a Dataset of Real and Synthesized Guitar Recordings

- Authors: Marios Glytsos, Christos Garoufis, Athanasia Zlatintsi, Petros Maragos
- Conference: 25th International Society for Music Information Retrieval Conference (ISMIR 2024)

It supports three main jobs:

2. Train a model.
3. Run separation with a saved checkpoint.
4. Evaluate the separated files with SDR / SIR / ISR / SAR / SI-SDR.

## Install the environment

Create the Conda environment with:

```bash
conda env create -f environment.yml
conda activate guitarduets
```

## Dataset

The dataset can be found on Zenodo:

[GuitarDuets](https://zenodo.org/records/12802440)

Description:

GuitarDuets is a dataset of classical guitar duet recordings with MIDI annotations. It contains around three hours of real and synthesized classical guitar duet recordings. The synthesized duets include note-level MIDI annotations. The dataset was recorded with four different classical guitars to increase timbral diversity. Some tracks were replayed with different guitars for the same reason. The recordings were made in a quiet acoustically treated room using a pair of Presonus PM-2 microphones, one for each guitar. A dedicated leakage-free test set of 7 tracks was also recorded. The files are provided as 44.1 kHz, 16-bit stereo WAV files.


## Expected track format

For training and evaluation, the code expects the dataset to be organized by track.

Each track should be one folder.

Inside each track folder, the repo expects:

```text
Track Name/
├── mix.wav
├── guitar1.wav
├── guitar2.wav
└── notes.csv
```

`notes.csv` contains note annotations aligned with the audio.

The important columns are:

- `start_time`: the sample index where the note starts
- `end_time`: the sample index where the note ends
- `instrument`: which guitar plays the note
  - `1` means `guitar1`
  - `2` means `guitar2`
- `note`: the MIDI pitch number

The audio is sampled at `44100` Hz, so for example:

- `start_time = 44100` means the note starts at 1 second
- `end_time = 88200` means the note ends at 2 seconds

## Dataset split config

You first need to set the dataset paths in:

[configs/datasets/guitarrecordings.yaml](configs/datasets/guitarrecordings.yaml)

Example:

```yaml
name: guitarrecordings
splits:
  train: /absolute/path/to/GuitarRecordings/train
  valid: /absolute/path/to/GuitarRecordings/test # if no validation is provided the script falls back to random splitting of the training set
  test: /absolute/path/to/GuitarRecordings/test # this is only for separation, you should change that if you only need to do inference/evaluation
manifest_output: data/manifests/guitarrecordings.json
```

After you set these paths, run:

```bash
python scripts/build_metadata.py --config configs/datasets/guitarrecordings.yaml
```

This creates the manifest file used by the training, separation, and evaluation scripts.

## Experiment config

Then you need to set the experiment configuration, for example:

```yaml
run:
  name: train_guitarrecordings_time_freq
dataset:
  manifest: data/manifests/guitarrecordings.json
  train_split: train
  valid_split: valid
  test_split: test
  normalize: true
model:
  name: htdemucs
  kwargs:
    segment: 4
    time_conditioning: true
    freq_conditioning: true
    sources:
      - guitar1
      - guitar2
audio:
  segment_seconds: 4
```

These settings should be set correctly depending on the version of the model that you want to train or evaluate, and depending on which branch conditioning version you want to use.

## Train

Run:

```bash
python scripts/train.py --config configs/experiments/train_guitarrecordings_time_freq.yaml
```

## Separate

When you have a checkpoint model, run:

```bash
python scripts/separate.py --config configs/experiments/eval_guitarrecordings_time_freq.yaml --checkpoint artifacts/checkpoints/train_guitarrecordings_time_freq/best.pt
```

This will separate the tracks that are defined in your `test` split in the YAML file you used to build the metadata, and store the separated WAV files in `artifacts/`.

## Evaluate

Then run:

```bash
python scripts/evaluate.py --config configs/experiments/eval_guitarrecordings_time_freq.yaml --predictions artifacts/predictions/train_guitarrecordings_time_freq/best
```

This will calculate the metrics and write the results in `artifacts/`.


## Citation

If you use this work in your research, please cite:

BibTeX:

```bibtex
@inproceedings{glytsos2024guitarduets,
  author = {Glytsos, Marios and Garoufis, Christos and Zlatintsi, Athanasia and Maragos, Petros},
  title = {Classical Guitar Duet Separation using GuitarDuets -- a Dataset of Real and Synthesized Guitar Recordings},
  booktitle = {Proceedings of the 25th International Society for Music Information Retrieval Conference (ISMIR)},
  year = {2024},
  month = nov,
  pages = {95--102}
}
```


THIRD-PARTY NOTICES
===================

This repository contains code adapted from or copied from the following projects:

Demucs (https://github.com/facebookresearch/demucs)
Copyright (c) Facebook, Inc. and its affiliates.
Licensed under the MIT License.