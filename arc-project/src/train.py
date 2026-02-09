import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import os
import random
from PIL import Image

# --- CONFIG ---
DATA_DIR = "dataset/processed"
MODEL_SAVE_PATH = "retail_model.pth"
BATCH_SIZE = 16  # Increased batch size for stable gradients
EPOCHS = 15      # Increased epochs for the harder task
LEARNING_RATE = 0.0005 # Lower LR for fine-tuning
# --------------

class RandomBackground(object):
    def __init__(self, bg_dir="dataset/backgrounds"):
        self.bg_images = []
        if os.path.exists(bg_dir):
            self.bg_images = [os.path.join(bg_dir, f) for f in os.listdir(bg_dir) if f.endswith(('.jpg', '.png'))]

    def __call__(self, img):
        if img.mode == 'RGBA':
            # 50% chance: Solid Color (Keep this, it's good for robustness)
            if not self.bg_images or random.random() < 0.5:
                r, g, b = [random.randint(0, 255) for _ in range(3)]
                bg = Image.new('RGB', img.size, (r, g, b))
            
            # 50% chance: Real Image Background
            else:
                bg_path = random.choice(self.bg_images)
                bg_tex = Image.open(bg_path).convert('RGB')
                bg_tex = bg_tex.resize(img.size) # Stretch to fit
                bg = bg_tex
            
            bg.paste(img, mask=img.split()[3])
            return bg
        return img.convert('RGB')

def train():
    print(f"[INFO] Checking dataset in {DATA_DIR}...")
    
    # 1. Advanced Transformations
    data_transforms = transforms.Compose([
        RandomBackground(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 2. Load Data
    try:
        full_dataset = datasets.ImageFolder(DATA_DIR, transform=data_transforms)
    except FileNotFoundError:
        print("Error: Dataset not found.")
        return

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    class_names = full_dataset.classes
    print(f"[INFO] Classes: {class_names}")
    print(f"[INFO] Training on {len(train_dataset)} images | Validation on {len(val_dataset)}")

    # 3. Setup Model
    print("[INFO] Loading EfficientNet-B0...") 
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    
    # Get the correct input features for the classifier (which is 1280 for B0)
    # The classifier in torchvision's EfficientNet is a Sequential: [Dropout, Linear]
    num_features = model.classifier[1].in_features 
    model.classifier[1] = nn.Linear(num_features, len(class_names))
    
    # Strategy: Freeze first 80% of layers, Train last 20%
    total_layers = len(list(model.features.children()))
    freeze_until = int(total_layers * 0.8)
    
    for i, child in enumerate(model.features.children()):
        if i < freeze_until:
            for param in child.parameters():
                param.requires_grad = False
        else:
            for param in child.parameters():
                param.requires_grad = True

    # --- REMOVED THE DUPLICATE BLOCK THAT WAS CAUSING THE ERROR ---

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    # Only optimize parameters that have grad enabled (the unfrozen 20% + classifier)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    # 4. Training Loop
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
        
        # Validation Step (Crucial for knowing real performance)
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

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    # Save classes
    with open("classes.txt", "w") as f:
        f.write("\n".join(class_names))
        
    print(f"[DONE] Best Model Saved (Val Acc: {best_acc:.1f}%)")

if __name__ == "__main__":
    train()
