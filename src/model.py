"""
model.py — TissueClassifier definition and instantiation of both experiments.

Defines TissueClassifier, a ResNet-50 backbone with a configurable freeze flag
and a 9-class linear head that replaces the original 1000-class ImageNet head.

Two pre-built model instances are also exposed at module level:
  - model_a — frozen backbone (Exp A, linear probe baseline)
  - model_b — fully unfrozen backbone (Exp B, full fine-tuning)

Importing this module instantiates both models and downloads ImageNet weights
on first use (cached afterward). Both models are moved to the configured
device (CUDA if available, else CPU).
"""

import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

from src.config import NUM_CLASSES, DROPOUT_P, device


class TissueClassifier(nn.Module):
    """ResNet-50 with ImageNet-pretrained backbone and a 9-class head.

    Parameters
    ----------
    num_classes : int
        Number of output classes (default: 9, matching the 9 tissue types
        in NCT-CRC-HE-100K).
    freeze_backbone : bool
        If True (Exp A), all backbone parameters have requires_grad=False
        and only the 9-class head is trained. If False (Exp B), the entire
        network is fine-tunable.
    dropout_p : float
        Dropout probability applied before the final linear layer.
        Acts as regularization on the 9-class head.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, freeze_backbone: bool = True, dropout_p: float = DROPOUT_P):
        super().__init__()
        backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)

        # Freeze all backbone weights for Experiment A (linear probe)
        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

        # Remove original 1000-class head, keep everything up to (and including)
        # the global average pooling layer that produces a 2048-dim feature vector.
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        # 9-class head: flatten pooled features -> dropout -> linear
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_p),
            nn.Linear(2048, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


# --- Pre-built model instances for the two experiments ---
# Both share the same architecture; only the freeze_backbone flag differs.
model_a = TissueClassifier(freeze_backbone=True).to(device)   # Exp A — frozen backbone
model_b = TissueClassifier(freeze_backbone=False).to(device)  # Exp B — full fine-tuning