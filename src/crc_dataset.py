"""
crc_dataset.py — DataLoader construction for NCT-CRC-HE-100K and CRC-VAL-HE-7K.

Builds three DataLoaders that the training and evaluation scripts consume:
  - train_loader  — 80% of NCT-CRC-HE-100K, with augmentation
  - val_loader    — 20% of NCT-CRC-HE-100K, no augmentation (clean images)
  - test_loader   — entire CRC-VAL-HE-7K, held out until final evaluation

Important behavior — module-level execution:
  Importing this module triggers ImageFolder construction and the stratified
  train/val split immediately. This means any script that imports from this
  module (including evaluate.py) will pause briefly while the loaders are built,
  even if it only needs test_loader. Acceptable for a course project; for
  larger codebases the pro pattern is to wrap construction in functions like
  get_test_loader() so only the needed loaders are built.

The stratified split preserves per-class proportions across train and val,
which matters because the dataset has moderate imbalance (~1.6× difference
between the largest and smallest classes).
"""

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from src.config import IMAGENET_MEAN, IMAGENET_STD, DATA_DIR, SEED, BATCH_SIZE


# --- Transforms ---
# Augmentation applied only to training data. Val and test see clean, normalized
# images so that evaluation metrics reflect model performance on real samples,
# not on an arbitrary augmentation policy.
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# --- Full training dataset (no transform applied yet) ---
# We build the dataset without a transform, then assign the appropriate
# transform per split via TransformSubset (defined below). This is necessary
# because train and val need different transforms but share the same
# underlying ImageFolder.
full_dataset = datasets.ImageFolder(root=DATA_DIR + "NCT-CRC-HE-100K")


class TransformSubset(Dataset):
    """Wraps a Subset to apply a transform on __getitem__.

    PyTorch's Subset doesn't support per-split transforms natively. This
    wrapper lets us assign train_transforms to the train indices and
    val_transforms to the val indices, while sharing the underlying
    ImageFolder.
    """

    def __init__(self, subset: Subset, transform: transforms.Compose):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        return self.transform(image), label

    def __len__(self) -> int:
        return len(self.subset)


# --- Stratified 80/20 split ---
# Stratification preserves the per-class proportions in both splits, so val
# accuracy reflects the same class distribution as training rather than being
# biased by random sampling.
labels  = [label for _, label in full_dataset.samples]
indices = list(range(len(full_dataset)))

train_idx, val_idx = train_test_split(
    indices,
    test_size=0.2,
    stratify=labels,
    random_state=SEED,
)

train_dataset = TransformSubset(Subset(full_dataset, train_idx), train_transforms)
val_dataset   = TransformSubset(Subset(full_dataset, val_idx),   val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)


# --- Held-out test set ---
# Never touched until final evaluation in evaluate.py. Uses the val transforms
# (no augmentation, just normalization) since this is inference-only.
test_dataset = datasets.ImageFolder(root=DATA_DIR + "CRC-VAL-HE-7K", transform=val_transforms)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)