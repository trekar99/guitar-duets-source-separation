# Guitar Duets Source Separation

Monotimbral music source separation for classical guitar duets, building on [GuitarDuets](https://zenodo.org/records/12802440) (ISMIR 2024).

> **Classical Guitar Duet Separation using GuitarDuets — a Dataset of Real and Synthesized Guitar Recordings**
> Glytsos, Garoufis, Zlatintsi, Maragos — ISMIR 2024

## Project Structure

```
├── configs/
│   ├── dataset.yaml          # Dataset paths (edit for your machine)
│   ├── conditioned.yaml      # Score-conditioned experiment
│   └── unconditioned.yaml    # Unconditioned experiment
├── dataset/                  # Audio data (Real/ and Synth/, gitignored)
├── manifests/                # Generated manifest JSONs (gitignored)
├── outputs/                  # All outputs (gitignored)
│   ├── checkpoints/          # Model weights
│   ├── predictions/          # Separated audio
│   ├── metrics/              # Evaluation results
│   ├── logs/                 # Training history
│   └── plots/                # Visualizations
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_training_demo.ipynb
│   ├── 03_separation_demo.ipynb
│   ├── 04_evaluation_analysis.ipynb
│   ├── 05_basic_pitch_splitting.ipynb    # Improvement 1
│   ├── 06_midi_guided_masking.ipynb      # Improvement 2
│   └── 07_combined_pipeline.ipynb        # All improvements combined
├── scripts/                  # CLI entry points
└── src/                      # Core library
    ├── data/                 # Dataset, manifests, metadata
    ├── evaluation/           # BSS-eval metrics, SI-SDR
    ├── inference/            # Full-track separation
    ├── models/               # HTDemucs, HDemucs, Demucs
    ├── plotting/             # Visualizations
    ├── training/             # Engine, augmentations, losses
    └── utils/                # Audio I/O, config, helpers
```

## Quick Start

### 1. Install

```bash
conda env create -f environment.yml
conda activate guitarduets
```

### 2. Configure dataset paths

Edit `configs/dataset.yaml` with your local paths, then build the manifest:

```bash
python scripts/build_metadata.py --config configs/dataset.yaml
```

### 3. Train

```bash
# Unconditioned
python scripts/train.py --config configs/unconditioned.yaml

# Score-conditioned
python scripts/train.py --config configs/conditioned.yaml

# Fine-tune from checkpoint
python scripts/train.py --config configs/conditioned.yaml --checkpoint outputs/checkpoints/best_conditioned.pt
```

### 4. Separate & Evaluate

```bash
python scripts/run_pipeline.py --config configs/unconditioned.yaml --checkpoint outputs/checkpoints/best_unconditioned.pt
```

Or separately:

```bash
python scripts/separate.py --config configs/unconditioned.yaml --checkpoint outputs/checkpoints/best_unconditioned.pt
python scripts/evaluate.py --config configs/unconditioned.yaml --predictions outputs/predictions/unconditioned
```

## Training Features

| Feature | Details |
|---|---|
| Mixed Precision (AMP) | Enabled by default on CUDA |
| Cosine Annealing LR | Smooth decay over training schedule |
| Gradient Clipping | Max norm 5.0 (configurable) |
| Early Stopping | Configurable patience |
| PIT Loss | Permutation-invariant L1 via asteroid |
| Score-Informed Augmentation | Joint audio+notes shifts, note jitter |

## Improvements

| # | Improvement | Type | Retraining? |
|---|---|---|---|
| 1 | Basic Pitch + Heuristic Pitch Splitting | Inference optimization | No |
| 2 | MIDI-Guided Post-Processing Audio Masking | Post-processing | No |
| 3 | Misaligned Data Augmentation & Synchronized Transforms | Training augmentation | Yes (fine-tuning) |

## Citation

```bibtex
@inproceedings{glytsos2024guitarduets,
  author    = {Glytsos, Marios and Garoufis, Christos and Zlatintsi, Athanasia and Maragos, Petros},
  title     = {Classical Guitar Duet Separation using {GuitarDuets}},
  booktitle = {Proc. ISMIR},
  year      = {2024},
  pages     = {95--102}
}
```

## Third-Party Notices

Contains code adapted from [Demucs](https://github.com/facebookresearch/demucs) (Facebook Research, MIT License).
