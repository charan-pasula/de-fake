import torch
import torch.nn as nn
import torchvision.models as models

class VisionTransformer(nn.Module):
    def __init__(self, num_classes=2, freeze_backbone=False):
        super().__init__()
        # Load pre-trained ViT
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        
        if freeze_backbone:
            # Freeze all parameters so we don't spend CPU time updating them
            for param in self.vit.parameters():
                param.requires_grad = False
                
        # Replace the head for our specific number of classes (FAKE/REAL)
        in_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        # The torchvision ViT just returns logits
        logits = self.vit(x)
        
        # To remain compatible with scripts that expect (logits, attention)
        # We return None for attention
        return logits, None