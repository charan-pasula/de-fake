import torch
import torch.nn as nn
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from model import VisionTransformer

def main():
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), # Scaling
        transforms.RandomHorizontalFlip(),                   # Flipping
        transforms.RandomRotation(15),                       # Rotation
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Color Jittering
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Dataset without transforms to split first
    data_dir = "Data"
    if not os.path.exists(data_dir):
        print(f"Error: Directory '{data_dir}' not found.")
        return

    full_dataset = datasets.ImageFolder(data_dir)
    
    print("Total Images:", len(full_dataset))
    print("Classes:", full_dataset.classes)

    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_subset, val_subset, test_subset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    # Custom Dataset class to apply transforms to subsets
    class TransformDataset(torch.utils.data.Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform

        def __getitem__(self, index):
            x, y = self.subset[index]
            if self.transform:
                x = self.transform(x)
            return x, y

        def __len__(self):
            return len(self.subset)

    train_dataset = TransformDataset(train_subset, train_transform)
    val_dataset = TransformDataset(val_subset, val_transform)
    test_dataset = TransformDataset(test_subset, val_transform)

    print("Train Images:", len(train_dataset))
    print("Validation Images:", len(val_dataset))
    print("Test Images:", len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = VisionTransformer().to(device)
    
    # Check if a saved model exists to resume training
    model_path = "saved_model/deepfake_vit.pth"
    if os.path.exists(model_path):
        print(f"\n[INFO] Found existing model at {model_path}. Loading weights to resume training...\n")
        model.load_state_dict(torch.load(model_path, map_location=device))
        
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 10
    best_val_loss = float('inf')
    
    os.makedirs("saved_model", exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        print(f"\nStarting Epoch {epoch+1}/{epochs}")

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Batch: {batch_idx}, Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs, _ = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        if len(val_loader) > 0:
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = 100 * correct / total
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Val Accuracy: {val_accuracy:.2f}%")
        else:
            avg_val_loss = avg_train_loss # Fallback if val set is empty
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} (No validation data)")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "saved_model/deepfake_vit.pth")
            print("Model improved! Saved successfully.")

if __name__ == "__main__":
    main()