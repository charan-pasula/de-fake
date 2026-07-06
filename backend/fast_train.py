import torch
import torch.nn as nn
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from backend.model import VisionTransformer

def main():
    print("Initializing fast training (using 200 images for a quick demo)...")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Data")
    full_dataset = datasets.ImageFolder(data_dir, transform=transform)
    
    # The dataset has 3810 FAKE images and 3825 REAL images.
    # We take the first 100 FAKE and the first 100 REAL images to train instantly.
    fake_indices = list(range(0, 100))
    real_indices = list(range(3810, 3910))
    subset_indices = fake_indices + real_indices
    
    train_dataset = Subset(full_dataset, subset_indices)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = VisionTransformer().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    epochs = 2 # 2 quick epochs
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}...")
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, _ = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 5 == 0:
                print(f"  Batch {batch_idx} Loss: {loss.item():.4f}")

    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "saved_model")
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "deepfake_vit.pth"))
    print("\nFast training complete! Model weights saved.")
    print("You can now run 'python app.py' to test the predictions.")

if __name__ == "__main__":
    main()
