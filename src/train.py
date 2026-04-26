"""
train.py — Training loop for both transfer-learning experiments.

Defines three functions:
  - train_one_epoch(): one full pass over the training set with backprop
  - validate():        one full pass over the validation set, no gradients
  - run_experiment():  full multi-epoch training run with checkpointing

run_experiment() is the entry point used by the training notebook. It saves
the best checkpoint by validation accuracy to outputs/checkpoints/ and
returns the full training history along with timing and peak-memory stats.

Note: history dicts returned by run_experiment() should be saved to disk by
the caller (np.save with allow_pickle=True). The training notebook in this
project did not save them, which required reconstruction via
extract_training_history.py — a lesson learned for future runs.
"""

import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.crc_dataset import train_loader, val_loader, test_loader
from src.config      import BATCH_SIZE, NUM_CLASSES, NUM_EPOCHS, device, CHECKPOINT_DIR


def train_one_epoch(model, loader, optimizer, criterion, device):
    """One pass over the training set with backprop and parameter updates.

    Returns
    -------
    avg_loss : float
        Sum of batch losses divided by total samples (not number of batches).
    accuracy : float
        Fraction of training samples correctly predicted this epoch.
    """
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    return running_loss / total, correct / total


def validate(model, loader, criterion, device):
    """One pass over the validation set with gradients disabled.

    Returns
    -------
    avg_loss : float
        Sum of batch losses divided by total samples.
    accuracy : float
        Fraction of validation samples correctly predicted.
    """
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            running_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    return running_loss / total, correct / total


def run_experiment(model, train_loader, val_loader, optimizer, scheduler=None,
                   num_epochs=NUM_EPOCHS, experiment_name="Exp"):
    """Full multi-epoch training run with best-checkpoint saving.

    Parameters
    ----------
    model : nn.Module
        The model to train (e.g., TissueClassifier instance).
    train_loader, val_loader : DataLoader
        Training and validation DataLoaders.
    optimizer : torch.optim.Optimizer
        Pre-configured optimizer. For Exp B this includes differential LRs
        across parameter groups (backbone vs head).
    scheduler : torch.optim.lr_scheduler, optional
        LR scheduler stepped once per epoch. None for Exp A; cosine annealing
        for Exp B.
    num_epochs : int
        Total training epochs (default: NUM_EPOCHS from config).
    experiment_name : str
        Used as a tag in print logs and as the prefix for the saved
        checkpoint filename ({experiment_name}_best.pth).

    Returns
    -------
    history : dict
        Per-epoch lists of train_loss, train_acc, val_loss, val_acc.
    best_val_acc : float
        Best validation accuracy achieved across all epochs.
    elapsed : float
        Total training time in seconds.
    peak_mem : float
        Peak CUDA memory allocated during the run, in megabytes.
    """
    criterion = nn.CrossEntropyLoss()
    history = {
        "train_loss": [],
        "train_acc":  [],
        "val_loss":   [],
        "val_acc":    [],
    }

    best_val_acc = 0.0

    start = time.time()
    torch.cuda.reset_peak_memory_stats()

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss,   val_acc   = validate(model, val_loader, criterion, device)

        if scheduler:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Save best checkpoint by val accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{CHECKPOINT_DIR}{experiment_name}_best.pth")

        print(f"[{experiment_name}] Epoch {epoch + 1}/{num_epochs} | "
              f"Train loss: {train_loss:.4f} acc: {train_acc:.4f} | "
              f"Val loss: {val_loss:.4f} acc: {val_acc:.4f}")

    elapsed  = time.time() - start
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2

    return history, best_val_acc, elapsed, peak_mem