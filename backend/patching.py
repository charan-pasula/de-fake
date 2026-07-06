import torch
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

print("Original Shape:", image.shape)

patch_size = 16

patches = image.unfold(2, patch_size, patch_size)\
               .unfold(3, patch_size, patch_size)

print("Patch Shape:", patches.shape)