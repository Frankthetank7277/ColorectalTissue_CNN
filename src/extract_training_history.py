"""
extract_training_history.py — One-time utility to reconstruct training history dicts
from the printed epoch logs in 02_training.ipynb.

Run once from the project root:
    python -m src.scrape_history
"""

import re
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR  = PROJECT_ROOT / "outputs" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


EXP_A_LOG = """
[Exp_A] Epoch 1/20 | Train loss: 0.0116 acc: 0.8903 | Val loss: 0.0053 acc: 0.9535
[Exp_A] Epoch 2/20 | Train loss: 0.0080 acc: 0.9173 | Val loss: 0.0046 acc: 0.9583
[Exp_A] Epoch 3/20 | Train loss: 0.0079 acc: 0.9163 | Val loss: 0.0043 acc: 0.9601
[Exp_A] Epoch 4/20 | Train loss: 0.0077 acc: 0.9179 | Val loss: 0.0040 acc: 0.9615
[Exp_A] Epoch 5/20 | Train loss: 0.0077 acc: 0.9184 | Val loss: 0.0042 acc: 0.9599
[Exp_A] Epoch 6/20 | Train loss: 0.0077 acc: 0.9184 | Val loss: 0.0038 acc: 0.9637
[Exp_A] Epoch 7/20 | Train loss: 0.0078 acc: 0.9174 | Val loss: 0.0039 acc: 0.9619
[Exp_A] Epoch 8/20 | Train loss: 0.0077 acc: 0.9186 | Val loss: 0.0038 acc: 0.9634
[Exp_A] Epoch 9/20 | Train loss: 0.0077 acc: 0.9193 | Val loss: 0.0037 acc: 0.9636
[Exp_A] Epoch 10/20 | Train loss: 0.0076 acc: 0.9208 | Val loss: 0.0038 acc: 0.9635
[Exp_A] Epoch 11/20 | Train loss: 0.0077 acc: 0.9190 | Val loss: 0.0037 acc: 0.9649
[Exp_A] Epoch 12/20 | Train loss: 0.0076 acc: 0.9193 | Val loss: 0.0036 acc: 0.9648
[Exp_A] Epoch 13/20 | Train loss: 0.0076 acc: 0.9200 | Val loss: 0.0038 acc: 0.9638
[Exp_A] Epoch 14/20 | Train loss: 0.0076 acc: 0.9188 | Val loss: 0.0037 acc: 0.9649
[Exp_A] Epoch 15/20 | Train loss: 0.0076 acc: 0.9193 | Val loss: 0.0037 acc: 0.9639
[Exp_A] Epoch 16/20 | Train loss: 0.0076 acc: 0.9200 | Val loss: 0.0038 acc: 0.9627
[Exp_A] Epoch 17/20 | Train loss: 0.0076 acc: 0.9203 | Val loss: 0.0036 acc: 0.9655
[Exp_A] Epoch 18/20 | Train loss: 0.0076 acc: 0.9202 | Val loss: 0.0037 acc: 0.9633
[Exp_A] Epoch 19/20 | Train loss: 0.0077 acc: 0.9199 | Val loss: 0.0038 acc: 0.9632
[Exp_A] Epoch 20/20 | Train loss: 0.0077 acc: 0.9196 | Val loss: 0.0037 acc: 0.9644
"""

EXP_B_LOG = """
[Exp_B] Epoch 1/20 | Train loss: 0.0105 acc: 0.9038 | Val loss: 0.0020 acc: 0.9809
[Exp_B] Epoch 2/20 | Train loss: 0.0029 acc: 0.9711 | Val loss: 0.0012 acc: 0.9883
[Exp_B] Epoch 3/20 | Train loss: 0.0020 acc: 0.9799 | Val loss: 0.0007 acc: 0.9930
[Exp_B] Epoch 4/20 | Train loss: 0.0014 acc: 0.9852 | Val loss: 0.0006 acc: 0.9944
[Exp_B] Epoch 5/20 | Train loss: 0.0011 acc: 0.9889 | Val loss: 0.0005 acc: 0.9949
[Exp_B] Epoch 6/20 | Train loss: 0.0009 acc: 0.9909 | Val loss: 0.0004 acc: 0.9963
[Exp_B] Epoch 7/20 | Train loss: 0.0007 acc: 0.9926 | Val loss: 0.0003 acc: 0.9963
[Exp_B] Epoch 8/20 | Train loss: 0.0006 acc: 0.9937 | Val loss: 0.0003 acc: 0.9961
[Exp_B] Epoch 9/20 | Train loss: 0.0005 acc: 0.9945 | Val loss: 0.0003 acc: 0.9970
[Exp_B] Epoch 10/20 | Train loss: 0.0005 acc: 0.9952 | Val loss: 0.0003 acc: 0.9967
[Exp_B] Epoch 11/20 | Train loss: 0.0004 acc: 0.9960 | Val loss: 0.0003 acc: 0.9970
[Exp_B] Epoch 12/20 | Train loss: 0.0003 acc: 0.9964 | Val loss: 0.0002 acc: 0.9975
[Exp_B] Epoch 13/20 | Train loss: 0.0003 acc: 0.9964 | Val loss: 0.0002 acc: 0.9978
[Exp_B] Epoch 14/20 | Train loss: 0.0003 acc: 0.9973 | Val loss: 0.0002 acc: 0.9977
[Exp_B] Epoch 15/20 | Train loss: 0.0003 acc: 0.9972 | Val loss: 0.0002 acc: 0.9980
[Exp_B] Epoch 16/20 | Train loss: 0.0002 acc: 0.9974 | Val loss: 0.0002 acc: 0.9978
[Exp_B] Epoch 17/20 | Train loss: 0.0002 acc: 0.9976 | Val loss: 0.0002 acc: 0.9981
[Exp_B] Epoch 18/20 | Train loss: 0.0002 acc: 0.9979 | Val loss: 0.0002 acc: 0.9979
[Exp_B] Epoch 19/20 | Train loss: 0.0002 acc: 0.9979 | Val loss: 0.0002 acc: 0.9981
[Exp_B] Epoch 20/20 | Train loss: 0.0002 acc: 0.9976 | Val loss: 0.0002 acc: 0.9983
"""

PATTERN = re.compile(
    r"\[(?P<exp>Exp_[AB])\]\s+"
    r"Epoch\s+(?P<epoch>\d+)/\d+\s+\|\s+"
    r"Train loss:\s+(?P<train_loss>[\d.]+)\s+"
    r"acc:\s+(?P<train_acc>[\d.]+)\s+\|\s+"
    r"Val loss:\s+(?P<val_loss>[\d.]+)\s+"
    r"acc:\s+(?P<val_acc>[\d.]+)"
)


def parse_log(log: str) -> dict:
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    for line in log.strip().splitlines():
        m = PATTERN.search(line)
        if m is None:
            continue
        history["train_loss"].append(float(m.group("train_loss")))
        history["train_acc"].append(float(m.group("train_acc")))
        history["val_loss"].append(float(m.group("val_loss")))
        history["val_acc"].append(float(m.group("val_acc")))
    return history


def main():
    history_a = parse_log(EXP_A_LOG)
    history_b = parse_log(EXP_B_LOG)

    print(f"Exp A epochs parsed: {len(history_a['train_loss'])}")
    print(f"Exp B epochs parsed: {len(history_b['train_loss'])}")
    print(f"Exp A best val acc:  {max(history_a['val_acc'])}")
    print(f"Exp B best val acc:  {max(history_b['val_acc'])}")

    np.save(RESULTS_DIR / "Exp_A_history.npy", history_a, allow_pickle=True)
    np.save(RESULTS_DIR / "Exp_B_history.npy", history_b, allow_pickle=True)

    print(f"\nSaved to {RESULTS_DIR}:")
    print("  Exp_A_history.npy")
    print("  Exp_B_history.npy")


if __name__ == "__main__":
    main()