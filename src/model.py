# model.py
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights
from src.config import NUM_CLASSES, DROPOUT_P, device

class TissueClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, freeze_backbone=True, dropout_p=DROPOUT_P):
        super().__init__()
        backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)

        # Freeze all backbone weights for Experiment A (linear probe)
        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

        # Remove original 1000-class head, keep feature extractor
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        # 9-class head: flatten pooled features → dropout → linear
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_p),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

# Experiment A — frozen backbone, head only
model_a = TissueClassifier(freeze_backbone=True).to(device)

# Experiment B — full fine-tuning with differential LRs
model_b = TissueClassifier(freeze_backbone=False).to(device)