# ColorectalTissue_CNN

Transfer learning with ResNet for 9-class colorectal tissue classification on H&E-stained histopathology images.

## Results Summary

Fine-tuned ResNet achieves **95.19% accuracy** (macro F1 = 0.935) on the CRC-VAL-HE-7K external test set (n=7,180). Comparison of two training strategies demonstrates the value of end-to-end fine-tuning over frozen feature extraction:

| Experiment | Strategy | Test Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|---|
| Exp_A | Frozen backbone, trainable classification head | 91.66% | 0.8961 | 0.9150 |
| Exp_B | Full fine-tuning, differential LRs + cosine annealing | **95.19%** | **0.9353** | **0.9512** |

**Key finding:** Full fine-tuning with differential learning rates (lower LR for pretrained backbone, higher LR for new classification head) yielded a +3.53% accuracy improvement over frozen feature extraction, confirming that adapting pretrained representations to domain-specific imagery outperforms fixed-feature transfer learning for histopathology.

## Project Overview

This project investigates two transfer learning strategies for multi-class tissue classification on hematoxylin & eosin (H&E)-stained colorectal histopathology images. The experiments isolate the effect of backbone adaptation: Experiment A treats the pretrained ResNet as a fixed feature extractor, while Experiment B fine-tunes the full network with carefully tuned differential learning rates and a cosine annealing schedule.

**Course:** BIOE 486 — Applied Deep Learning, UIUC (Spring 2026)
**Author:** Frank Lato

## Dataset

**[NCT-CRC-HE-100K](https://zenodo.org/records/1214456)** (Kather et al., 2018) — 100,000 non-overlapping 224×224 H&E-stained image patches from 86 whole-slide images, Macenko color-normalized, spanning 9 tissue classes.

**[CRC-VAL-HE-7K](https://zenodo.org/records/1214456)** — 7,180 patches from 50 patients (no overlap with NCT-CRC-HE-100K) used as the external validation set.

**Tissue classes (9):**
- ADI — Adipose tissue
- BACK — Background
- DEB — Debris
- LYM — Lymphocytes
- MUC — Mucus
- MUS — Smooth muscle
- NORM — Normal colon mucosa
- STR — Cancer-associated stroma
- TUM — Colorectal adenocarcinoma epithelium

## Methodology

1. **Dataset preparation** — Custom PyTorch `Dataset` wrapping the Kather et al. patches with standard image normalization and augmentation
2. **Model architecture** — Pretrained ResNet backbone with custom classification head for 9 classes
3. **Experiment A — Frozen feature extraction** — All backbone parameters frozen; only classification head trained with Adam
4. **Experiment B — Full fine-tuning** — All layers unfrozen with differential learning rates (backbone LR < head LR) and cosine annealing scheduler to stabilize training
5. **Evaluation** — Accuracy, macro/weighted F1, per-class precision/recall, and normalized/raw confusion matrices on CRC-VAL-HE-7K

## Repository Structure
ColorectalTissue_CNN/
├── src/
│   ├── config.py                       # Hyperparameters and paths
│   ├── crc_dataset.py                  # Custom PyTorch Dataset
│   ├── model.py                        # ResNet with custom head
│   ├── train.py                        # Training loop
│   ├── evaluate.py                     # Evaluation + metrics + plots
│   └── extract_training_history.py     # Pulls training curves from checkpoints
├── notebooks/
│   ├── 01_eda.ipynb                    # Exploratory data analysis
│   └── 02_training.ipynb               # End-to-end training pipeline
├── outputs/
│   ├── figures/                        # Confusion matrices, class samples
│   └── results/                        # Classification reports, history arrays
├── environment.yml                     # Conda environment spec (Python)
├── requirements.txt                    # Pip dependencies (PyTorch + libraries)
└── README.md

## Setup

This project uses a hybrid conda + pip environment. PyTorch is installed with CUDA 12.1 support for GPU training.

```bash
# Clone the repository
git clone git@github.com:Frankthetank7277/ColorectalTissue_CNN.git
cd ColorectalTissue_CNN

# Create conda environment (Python 3.11)
conda env create -f environment.yml
conda activate bioe486

# Install pip dependencies (PyTorch CUDA 12.1 + other libraries)
pip install -r requirements.txt
```

**GPU note:** `requirements.txt` pins `torch==2.5.1+cu121`. If you are on a different CUDA version or CPU-only, install the appropriate PyTorch build from [pytorch.org](https://pytorch.org/get-started/previous-versions/) before running `pip install -r requirements.txt`.

## Data Access

The NCT-CRC-HE-100K dataset (~15 GB) is not included in this repository. See [`data/README.md`](data/README.md) for download and setup instructions.

## Reproducing Results

1. Download and set up the dataset per `data/README.md`
2. Open `notebooks/02_training.ipynb`
3. Run all cells to reproduce both experiments end-to-end
4. Trained checkpoints save to `outputs/checkpoints/`; figures and reports save to `outputs/figures/` and `outputs/results/`

Training both experiments takes approximately 2 hours, 16 minutes on a single NVIDIA GPU.

## Status

🚧 Final project for BIOE 486 (Spring 2026). Will be made public after course completion.