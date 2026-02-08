import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import os
from PIL import Image

# --- CONFIG ---
DATA_DIR = "dataset/processed"
MODEL_SAVE_PATH = "retail_model.pth"
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 0.001
# --------------

def train():
    print(f"[INFO] Checking dataset in {DATA_DIR}...")
    
    # 1. Define Transformations
    # We must convert your Transparent PNGs (RGBA) to RGB (Black Background)
    # because Neural Networks expect 3 channels, not 4.
    class TransparentToBlack(object):
        def __call__(self, img):
            if img.mode == 'RGBA':
                # Create a black background
                bg = Image.new('RGB', img.size, (0, 0, 0))
                # Paste the image using alpha as mask
                bg.paste(img, mask=img.split()[3])
                return bg
            return img.convert('RGB')

    data_transforms = transforms.Compose([
        TransparentToBlack(),
        transforms.Resize((224, 224)),
        transforms.RandomRotation(20), # Add some variety
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 2. Load Data
    try:
        full_dataset = datasets.ImageFolder(DATA_DIR, transform=data_transforms)
    except FileNotFoundError:
        print("Error: Dataset not found. Are you running this from the 'arc-project' root?")
        return

    # Split: 80% Training, 20% Validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    class_names = full_dataset.classes
    print(f"[INFO] Detected Classes: {class_names}")
    print(f"[INFO] Training on {len(train_dataset)} images, Validating on {len(val_dataset)}")

    # 3. Setup Model (MobileNetV2)
    print("[INFO] Loading MobileNetV2 (Pre-trained)...")
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    
    # Freeze the early layers (so we don't destroy what Google learned)
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final classifier layer for YOUR classes
    model.classifier[1] = nn.Linear(model.last_channel, len(class_names))

    # Move to GPU if available (Arch Linux users often have CUDA setup)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training on: {device}")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)

    # 4. Training Loop
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

        epoch_acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss/len(train_loader):.4f} | Acc: {epoch_acc:.2f}%")

    # 5. Save
    print("[INFO] Saving model...")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    
    # Also save the class names to a text file for the backend to read
    with open("classes.txt", "w") as f:
        f.write("\n".join(class_names))
        
    print(f"[DONE] Model saved to {MODEL_SAVE_PATH}")
    print(f"[DONE] Classes saved to classes.txt")

if __name__ == "__main__":
    train()
