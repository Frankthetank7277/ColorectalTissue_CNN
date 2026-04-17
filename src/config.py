import os
import random
import numpy as np
import torch
from pathlib import Path

# Project root: .../Final_Project
ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = str(ROOT_DIR / "data") + "/"
CHECKPOINT_DIR = str(ROOT_DIR / "outputs/checkpoints") + "/"
FIGURES_DIR = str(ROOT_DIR / "outputs/figures") + "/"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42
BATCH_SIZE = 32
NUM_EPOCHS = 20
NUM_CLASSES = 9
DROPOUT_P = 0.5
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

LR_A = 1e-3
LR_B_BACKBONE = 1e-5
LR_B_HEAD = 1e-4

set_seed(SEED)
