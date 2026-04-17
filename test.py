import os
print("Working directory:", os.getcwd())
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = Path("data/NCT-CRC-HE-100K")
print("DATA_DIR exists:", DATA_DIR.exists())
print("Contents:", list(DATA_DIR.iterdir()) if DATA_DIR.exists() else "PATH NOT FOUND")
VAL_DIR  = Path("data/CRC-VAL-HE-7K")

# --- Class counts ---
print("=== NCT-CRC-HE-100K class distribution ===")
classes = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
counts = {}
for cls in classes:
    n = len(list((DATA_DIR / cls).glob("*.tif")))
    counts[cls] = n
    print(f"  {cls:<6}  {n:>6} images")
print(f"  {'TOTAL':<6}  {sum(counts.values()):>6} images")

print("\n=== CRC-VAL-HE-7K class distribution ===")
for cls in classes:
    n = len(list((VAL_DIR / cls).glob("*.tif")))
    print(f"  {cls:<6}  {n:>6} images")

# --- Spot check one image per class ---
fig, axes = plt.subplots(1, len(classes), figsize=(18, 3))
for ax, cls in zip(axes, classes):
    img_path = next((DATA_DIR / cls).glob("*.tif"))
    img = Image.open(img_path)
    ax.imshow(img)
    ax.set_title(cls, fontsize=9)
    ax.axis("off")
    print(f"  {cls}: size={img.size}, mode={img.mode}")
plt.tight_layout()
plt.savefig("outputs/figures/class_samples.png", dpi=150)
plt.show()