import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

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

plt.figure(figsize=(10,10))

count = 1

for i in range(14):
    for j in range(14):

        patch = image[
            0,
            :,
            i*16:(i+1)*16,
            j*16:(j+1)*16
        ]

        plt.subplot(14,14,count)

        plt.imshow(
            patch.permute(1,2,0)
        )

        plt.axis("off")

        count += 1

plt.show()