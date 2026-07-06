import torch
import torch.nn as nn
import matplotlib.pyplot as plt
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

# PATCHING
patches = image.unfold(2,16,16).unfold(3,16,16)

patches = patches.contiguous().view(
    1,
    196,
    768
)

# EMBEDDING
embedding = nn.Linear(768,768)
tokens = embedding(patches)

# Q K V
Q = nn.Linear(768,768)(tokens)
K = nn.Linear(768,768)(tokens)
V = nn.Linear(768,768)(tokens)

# ATTENTION
scores = torch.matmul(
    Q,
    K.transpose(-2,-1)
)

scores = scores / (768**0.5)

attention = torch.softmax(
    scores,
    dim=-1
)

# VISUALIZATION
heatmap = attention[0].detach().numpy()

plt.figure(figsize=(8,8))
plt.imshow(heatmap)
plt.colorbar()
plt.title("Attention Heatmap")
plt.show()