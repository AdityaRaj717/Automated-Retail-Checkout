import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import os
import random
from PIL import Image
from custom_model import RetailAttnNet

# --- CONFIG ---
DATA_DIR = "dataset/processed"
MODEL_SAVE_PATH = "retail_model.pth"
CLASS_FILE = "classes.txt"
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 0.0005
# --------------

class RandomBackground(object):
    """
    Pastes the transparent product image onto a random solid color background.
    This prevents the model from relying on the black void or specific edge artifacts.
    """
    def __call__(self, img):
        if img.mode == 'RGBA':
            # Generate a random color (R, G, B)
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            
            # Create a solid color background
            bg = Image.new('RGB', img.size, (r, g, b))
            
            # Paste the image using its own alpha channel as a mask
            bg.paste(img, mask=img.split()[3])
            return bg
        return img.convert('RGB')

def train():
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training on: {device}")
    
    print(f"[INFO] Checking dataset in {DATA_DIR}...")
    if not os.path.exists(DATA_DIR):
        print(f"[ERROR] Dataset directory '{DATA_DIR}' not found.")
        return

    # 2. Data Augmentation (The "Anti-Overfit" Suite)
    # We use random backgrounds + heavy distortion to make 40 images look like 4000
    data_transforms = transforms.Compose([
        RandomBackground(),  # Dynamic background generation
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Handles lighting changes
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),            # Handles angled views
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 3. Load Data
    try:
        full_dataset = datasets.ImageFolder(DATA_DIR, transform=data_transforms)
    except Exception as e:
        print(f"[ERROR] Could not load dataset: {e}")
        return

    # Save class names immediately so inference stays synced
    class_names = full_dataset.classes
    print(f"[INFO] Detected Classes: {class_names}")
    with open(CLASS_FILE, "w") as f:
        f.write("\n".join(class_names))

    # Split: 80% Train, 20% Val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"[INFO] Training on {len(train_dataset)} images, Validating on {len(val_dataset)}")

    # 4. Setup Model (Custom Research Architecture)
    print("[INFO] Loading Custom RetailAttnNet (Research Branch)...")
    
    model = RetailAttnNet(num_classes=len(class_names))
    
    # Move to GPU/CPU
    model = model.to(device)

    # NOTE: We do NOT freeze layers here. 
    # Since this is a custom model initialized with random weights, 
    # we must train ALL layers (Full Fine-Tuning) so they learn to recognize features.

    # 5. Training Setup
    criterion = nn.CrossEntropyLoss()
    # Only optimize parameters that require gradients
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    # 6. Training Loop
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%")

        if val_acc >= best_acc:
            best_acc = val_acc

    print(f"[INFO] Training finished. Saving final model state...")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"[DONE] Model saved to {MODEL_SAVE_PATH}")
    print(f"[INFO] Best Validation Accuracy observed: {best_acc:.1f}%")

if __name__ == "__main__":
    train()
