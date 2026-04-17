# Metrics & confusion matrix
"""
evaluate.py — Final evaluation on held-out CRC-VAL-HE-7K test set.

Loads the best checkpoints for Experiment A and Experiment B, runs inference
on the held-out test set, and produces:
  - Overall accuracy and macro-averaged F1
  - Confusion matrix (saved as .npy and as .png heatmap)
  - Per-class classification report (saved as .txt)
  - Side-by-side comparison of Exp A vs Exp B

Run from the project root:
    python -m src.evaluate
"""

# --- 1. Imports ---
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# Project imports
from src.config      import device
from src.model       import TissueClassifier
from src.crc_dataset import test_loader


# --- 2. Paths ---
# Anchor all paths to the project root (parent of src/), regardless of
# what directory the script is launched from.
from pathlib import Path

PROJECT_ROOT   = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
FIGURES_DIR    = PROJECT_ROOT / "outputs" / "figures"
RESULTS_DIR    = PROJECT_ROOT / "outputs" / "results"


# Make sure results dir exists (figures dir already exists from EDA)
os.makedirs(RESULTS_DIR, exist_ok=True)
# Class names — pulled from the test dataset so ordering matches label indices
CLASS_NAMES = test_loader.dataset.classes

# --- 3. Core evaluate function ---
def evaluate(model, test_loader, device, experiment_name):
    """
    Runs inference over the full test set.

    Returns
    -------
    y_true : np.ndarray of shape (n_samples,)
        Ground truth integer labels.
    y_pred : np.ndarray of shape (n_samples,)
        Predicted integer labels (argmax of logits).
    """
    # Set model to eval mode
    model.eval()

    all_preds  = []
    all_labels = []

    # No gradients needed during inference
    with torch.no_grad():
        for images, labels in test_loader:
            # Move batch to device
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            logits = model(images)

            # Convert logits to predicted class indices
            preds = logits.argmax(dim=1)

            # Move to CPU, convert to numpy, append to running lists
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(f"[{experiment_name}] Inference complete — {len(all_labels)} samples")

    return np.array(all_labels), np.array(all_preds)

# --- 4. Metrics + saving ---
def compute_and_save_metrics(y_true, y_pred, experiment_name):
    """
    Computes evaluation metrics, saves raw artifacts to disk, and returns
    a summary dict for the main comparison.

    Saves
    -----
    outputs/results/{experiment_name}_confusion_matrix.npy
    outputs/results/{experiment_name}_classification_report.txt

    Returns
    -------
    dict with keys: 'experiment', 'accuracy', 'f1_macro', 'cm'
    """
    # Compute core metrics
    # Reminder: sklearn convention is (y_true, y_pred) — do NOT swap
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    cm       = confusion_matrix(y_true, y_pred)

    # Per-class text report (precision, recall, F1, support)
    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,     # so the report shows class names, not indices 0-8
        digits=4,
    )

    # --- Save artifacts ---
    # Confusion matrix as .npy so it can be reloaded later without re-running inference
    np.save(f"{RESULTS_DIR}/{experiment_name}_confusion_matrix.npy", cm)

    # Classification report as .txt — human-readable for the appendix of your report
    report_path = f"{RESULTS_DIR}/{experiment_name}_classification_report.txt"
    with open(report_path, "w") as f:
        f.write(f"[{experiment_name}] Test-set results on CRC-VAL-HE-7K\n")
        f.write(f"Accuracy: {accuracy:.4f} | Macro F1: {f1_macro:.4f}\n\n")
        f.write(report)

    # Console summary
    print(f"[{experiment_name}] Accuracy: {accuracy:.4f} | Macro F1: {f1_macro:.4f}")
    print(f"  saved: {RESULTS_DIR}/{experiment_name}_confusion_matrix.npy")
    print(f"  saved: {report_path}")

    return {
        "experiment": experiment_name,
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "cm": cm,
    }

# --- 5. Confusion matrix heatmap ---
def plot_confusion_matrix(cm, class_names, experiment_name, normalize=True):
    """
    Renders the confusion matrix as a seaborn heatmap and saves to disk.

    Parameters
    ----------
    cm : np.ndarray of shape (n_classes, n_classes)
        Raw confusion matrix from sklearn.
    class_names : list of str
        Names to use as tick labels (same order as the rows/cols of cm).
    experiment_name : str
        Used in the title and in the saved filename.
    normalize : bool
        If True, display row-normalized values (fractions summing to 1 per row).
        If False, display raw counts.
    """
    if normalize:
        # Row-normalize: for each true class, what fraction went to each predicted class?
        cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
        title_suffix = " (row-normalized)"
    else:
        cm_display = cm
        fmt = "d"
        title_suffix = ""

    plt.figure(figsize=(9, 7))

    sns.heatmap(
        cm_display,                              # the matrix to display
        annot=True,                       # write the numeric value in each cell
        fmt=fmt,                          # formatting of annotations (".2f" or "d")
        cmap="Blues",
        xticklabels= class_names,                  # class names along x-axis
        yticklabels= class_names,                  # class names along y-axis
        cbar=True,
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{experiment_name} — Confusion Matrix{title_suffix}")
    plt.tight_layout()

    # Save — include "normalized" or "raw" in filename so both versions can coexist
    suffix = "normalized" if normalize else "raw"
    out_path = f"{FIGURES_DIR}/{experiment_name}_confusion_matrix_{suffix}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"  saved: {out_path}")

# --- 6. Main ---
def main():
    # Build fresh model instances
    # The freeze_backbone flag doesn't matter here — we're about to overwrite
    # all the weights with the checkpoint. But we should still match the
    # architecture that was used during training for clarity.
    model_a = TissueClassifier(freeze_backbone=True).to(device)
    model_b = TissueClassifier(freeze_backbone=False).to(device)

    # Load checkpoints from disk
    # map_location=device handles GPU/CPU portability
    ckpt_a_path = f"{CHECKPOINT_DIR}/Exp_A_best.pth"
    ckpt_b_path = f"{CHECKPOINT_DIR}/Exp_B_best.pth"

    model_a.load_state_dict(torch.load(ckpt_a_path, map_location=device, weights_only=True))
    model_b.load_state_dict(torch.load(ckpt_b_path, map_location=device, weights_only=True))

    print(f"Loaded checkpoint: {ckpt_a_path}")
    print(f"Loaded checkpoint: {ckpt_b_path}")
    print(f"Test set size: {len(test_loader.dataset)} samples across {len(CLASS_NAMES)} classes")
    print()

    # --- Experiment A ---
    print("=" * 60)
    print("Evaluating Experiment A (frozen backbone)")
    print("=" * 60)
    y_true_a, y_pred_a = evaluate(model_a, test_loader, device, "Exp_A")
    results_a = compute_and_save_metrics(y_true_a, y_pred_a, "Exp_A")
    plot_confusion_matrix(results_a["cm"], CLASS_NAMES, "Exp_A", normalize=True)
    plot_confusion_matrix(results_a["cm"], CLASS_NAMES, "Exp_A", normalize=False)

    # --- Experiment B ---
    print()
    print("=" * 60)
    print("Evaluating Experiment B (full fine-tuning)")
    print("=" * 60)
    y_true_b, y_pred_b = evaluate(model_b, test_loader, device, "Exp_B")
    results_b = compute_and_save_metrics(y_true_b, y_pred_b, "Exp_B")
    plot_confusion_matrix(results_b["cm"], CLASS_NAMES, "Exp_B", normalize=True)
    plot_confusion_matrix(results_b["cm"], CLASS_NAMES, "Exp_B", normalize=False)

    # --- Final side-by-side comparison ---
    print()
    print("=" * 60)
    print("Final Test-Set Comparison (CRC-VAL-HE-7K)")
    print("=" * 60)
    print(f"  Exp A — Accuracy: {results_a['accuracy']:.4f} | Macro F1: {results_a['f1_macro']:.4f}")
    print(f"  Exp B — Accuracy: {results_b['accuracy']:.4f} | Macro F1: {results_b['f1_macro']:.4f}")
    print(f"  Delta — Accuracy: {results_b['accuracy'] - results_a['accuracy']:+.4f} | "
          f"Macro F1: {results_b['f1_macro'] - results_a['f1_macro']:+.4f}")


if __name__ == "__main__":
    main()