# crc_dataset.py
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from src.config import IMAGENET_MEAN, IMAGENET_STD, DATA_DIR, SEED, BATCH_SIZE

# Augmentation for training only — val/test see clean images
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

# No transform applied here — assigned per split via TransformSubset
full_dataset = datasets.ImageFolder(root=DATA_DIR + "NCT-CRC-HE-100K")

# Wrapper to apply different transforms to train vs val subsets
class TransformSubset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        return self.transform(image), label

    def __len__(self):
        return len(self.subset)

# Stratified split to preserve class balance across train/val
labels = [label for _, label in full_dataset.samples]
indices = list(range(len(full_dataset)))

train_idx, val_idx = train_test_split(
    indices,
    test_size=0.2,
    stratify=labels,
    random_state=SEED
)

train_dataset = TransformSubset(Subset(full_dataset, train_idx), train_transforms)
val_dataset   = TransformSubset(Subset(full_dataset, val_idx),   val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# Held-out test set — not touched until final evaluation
test_dataset = datasets.ImageFolder(root=DATA_DIR + "CRC-VAL-HE-7K", transform=val_transforms)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)