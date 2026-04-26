"""
config.py — Central configuration for the tissue classification pipeline.

This module is the single source of truth for paths, hyperparameters, the
device handle, and reproducibility (seeds). All other modules import from
here rather than hardcoding values, ensuring consistency across the
training notebook, evaluation script, and EDA notebooks.

Importing this module also calls set_seed(SEED) at the bottom, so any
script that imports anything from src/ inherits a reproducible RNG state.
"""

import os
import random
import numpy as np
import torch
from pathlib import Path

# --- Paths ---
# Project root resolves to .../Final_Project regardless of where a script is launched from
ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR       = str(ROOT_DIR / "data") + "/"
CHECKPOINT_DIR = str(ROOT_DIR / "outputs/checkpoints") + "/"
FIGURES_DIR    = str(ROOT_DIR / "outputs/figures") + "/"


# --- Reproducibility ---
def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) for deterministic runs.

    Also disables cuDNN's nondeterministic kernels at the cost of some throughput.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Hyperparameters ---
SEED        = 42
BATCH_SIZE  = 32
NUM_EPOCHS  = 20
NUM_CLASSES = 9
DROPOUT_P   = 0.5
IMG_SIZE    = 224

# ImageNet normalization stats — required when using ImageNet-pretrained backbones
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Learning rates
# Exp A: single LR for the linear head (backbone frozen)
# Exp B: differential LRs — backbone fine-tunes slowly to preserve ImageNet features
LR_A          = 1e-3
LR_B_BACKBONE = 1e-5
LR_B_HEAD     = 1e-4

# Apply seed at import time so all downstream modules are reproducible
set_seed(SEED)
