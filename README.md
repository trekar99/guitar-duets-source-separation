# GuitarDuets Repository

This repo is the repo of the publication Classical Guitar Duet Separation using GuitarDuets -- a Dataset of Real and Synthesized Guitar Recordings

It supports four main jobs:

1. Build a manifest of the dataset.
2. Train a model.
3. Run separation with a saved checkpoint.
4. Evaluate the separated files with SDR / SIR / ISR / SAR / SI-SDR.

## Main folders

- `src/deguithybrid/`: the actual Python code
- `scripts/`: the commands you run
- `configs/`: the settings you edit
- `data/`: metadata and manifests
- `artifacts/`: outputs such as checkpoints, predictions, logs, and metrics
- `archive/`: old scripts and old experiment logs copied from the thesis code

## What the dataset should look like

For training and evaluation, the code expects the data to be arranged by track.

Each track should be one folder.

Inside each track folder, the repo expects:

```text
Track Name/
├── mix.wav
├── guitar1.wav
├── guitar2.wav
└── notes.csv
```

Training and separation both expect a `notes.csv` file in each track folder.

The required columns are:

- `start_time`
- `end_time`
- `instrument`
- `note`

Note start time and end time are in samples. 


## How the repo knows which files belong to train / valid / test

The repo does not guess this by itself.

You define it in:

[configs/datasets/guitarrecordings.yaml](/Users/mariosgly/Downloads/Python Scripts/GuitarDuets/deguithybridtrans/GuitarDuets/configs/datasets/guitarrecordings.yaml)

That file points to the root folders for each split.

Example:

```yaml
name: guitarrecordings
splits:
  train: /path/to/GuitarRecordings/train
  valid: /path/to/GuitarRecordings/valid
  test: /path/to/GuitarRecordings/test
manifest_output: data/manifests/guitarrecordings.json
```

If you do not define `valid`, the training script falls back to a random split from the training set.

## Files you edit before training

### 1. Dataset config

Edit:

[configs/datasets/guitarrecordings.yaml](/Users/mariosgly/Downloads/Python Scripts/GuitarDuets/deguithybridtrans/GuitarDuets/configs/datasets/guitarrecordings.yaml)

You change:

- where the train data lives
- where the valid data lives, if you have it
- where the test data lives

### 2. Experiment config

Edit:

[configs/experiments/train_guitarrecordings.yaml](/Users/mariosgly/Downloads/Python Scripts/GuitarDuets/deguithybridtrans/GuitarDuets/configs/experiments/train_guitarrecordings.yaml)

You change:

- the run name
- whether normalization is on
- which model to use
- segment length
- epochs
- batch size
- learning rate

## What happens before training

Before training, you run:

```bash
python scripts/build_metadata.py --config configs/datasets/guitarrecordings.yaml
```

This scans the dataset folders and creates a manifest JSON file.

That manifest contains:

- track name
- split
- path to the mix
- path to the two guitars
- sample rate
- track length
- mean
- standard deviation

The manifest is then used by the training and evaluation scripts.

## How to run training

From the reorganized repo folder, run:

```bash
python scripts/train.py --config configs/experiments/train_guitarrecordings.yaml
```

This will:

- load the manifest
- build the model
- create the dataloaders
- train the model
- save checkpoints
- save the training history

## Where training outputs go

Training outputs are saved here:

- checkpoints: `artifacts/checkpoints/<run_name>/`
- history/logs: `artifacts/logs/<run_name>/`

Example:

```text
artifacts/checkpoints/train_guitarrecordings/best.pt
artifacts/checkpoints/train_guitarrecordings/epoch_004.pt
artifacts/logs/train_guitarrecordings/history.json
```

## How to run separation later

Once you have a checkpoint, run:

```bash
python scripts/separate.py --config configs/experiments/eval_guitarrecordings.yaml --checkpoint artifacts/checkpoints/train_guitarrecordings/best.pt
```

This writes separated audio files to:

`artifacts/predictions/<run_name>/<checkpoint_name>/`

## How to run evaluation later

Once the predictions exist, run:

```bash
python scripts/evaluate.py --config configs/experiments/eval_guitarrecordings.yaml --predictions artifacts/predictions/train_guitarrecordings/best
```

This writes metrics to:

`artifacts/metrics/<run_name>/<checkpoint_name>/`

## Simple summary

If someone else uses this repo, the main thing they need to know is:

- put each song in its own folder
- each song folder must contain `mix`, `guitar1`, and `guitar2` audio files
- each song folder must also contain `notes.csv`
- set the dataset paths in the YAML config
- build the manifest
- train
- separate
- evaluate
