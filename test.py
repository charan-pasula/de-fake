import torch
import torchvision
import numpy
import PIL
import cv2

print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("NumPy:", numpy.__version__)
print("Pillow: Installed")
print("OpenCV:", cv2.__version__)

print("\nCUDA Available:", torch.cuda.is_available())