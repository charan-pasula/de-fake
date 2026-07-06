import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(
    "Data",
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True
)

image, label = next(iter(loader))

patch_size = 16

patches = image.unfold(2, patch_size, patch_size)\
               .unfold(3, patch_size, patch_size)

patches = patches.contiguous().view(
    1,
    196,
    3*16*16
)

print("Patch Tokens Shape:", patches.shape)

embedding = nn.Linear(
    768,
    768
)

embedded_patches = embedding(patches)

print("Embedded Shape:", embedded_patches.shape)