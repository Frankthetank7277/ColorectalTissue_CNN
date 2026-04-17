# train.py
import time
import torch
from torch import nn
from src.crc_dataset import train_loader, val_loader, test_loader
from src.config import BATCH_SIZE, NUM_CLASSES, NUM_EPOCHS, device, CHECKPOINT_DIR

def train_one_epoch(model, loader, optimizer, criterion, device):
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
        total += labels.size(0)

    return running_loss / total, correct / total

def validate(model, loader, criterion, device):
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
            total += labels.size(0)

    return running_loss / total, correct / total

def run_experiment(model, train_loader, val_loader, optimizer, scheduler=None, num_epochs=NUM_EPOCHS, experiment_name="Exp"):
    criterion = nn.CrossEntropyLoss()
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = 0.0

    start = time.time()
    torch.cuda.reset_peak_memory_stats()

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        if scheduler:
            scheduler.step()
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'{CHECKPOINT_DIR}{experiment_name}_best.pth')

        print(f"[{experiment_name}] Epoch {epoch + 1}/{num_epochs} | "
            f"Train loss: {train_loss:.4f} acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f} acc: {val_acc:.4f}")

    elapsed = time.time() - start
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2

    return history, best_val_acc, elapsed, peak_mem


