import torch.nn as nn
from torchvision import models

class DeepfakeModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = models.efficientnet_b0(weights="IMAGENET1K_V1")

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        in_features = self.backbone.classifier[1].in_features

        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.backbone(x)
