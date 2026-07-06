import torch
from model import VisionTransformer

model = VisionTransformer()

dummy_input = torch.randn(1, 3, 224, 224)

output, attention = model(dummy_input)

print("Output Shape:", output.shape)
if attention is not None:
    print("Attention Shape:", attention.shape)
else:
    print("Attention Shape: None (Using pre-trained ViT without attention hook)")