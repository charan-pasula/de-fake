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

# PATCHING
patch_size = 16

patches = image.unfold(2, patch_size, patch_size)\
               .unfold(3, patch_size, patch_size)

patches = patches.contiguous().view(
    1,
    196,
    768
)

# EMBEDDING
embedding = nn.Linear(768,768)
tokens = embedding(patches)

# Q K V
query = nn.Linear(768,768)
key   = nn.Linear(768,768)
value = nn.Linear(768,768)

Q = query(tokens)
K = key(tokens)
V = value(tokens)

print("Q Shape:", Q.shape)
print("K Shape:", K.shape)
print("V Shape:", V.shape)

# ATTENTION
scores = torch.matmul(
    Q,
    K.transpose(-2,-1)
)

scores = scores / (768 ** 0.5)

attention = torch.softmax(
    scores,
    dim=-1
)

print("Attention Shape:", attention.shape)

output = torch.matmul(
    attention,
    V
)

print("Output Shape:", output.shape)