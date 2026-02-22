"""
Metric Training Script — Triplet Loss Training for Few-Shot Recognition.

Replaces the standard CrossEntropy classifier training with a metric learning
approach using Triplet Margin Loss with online hard mining.

Usage:
    python metric_train.py                    # Full training from scratch
    python metric_train.py --fine-tune        # Fine-tune existing model
    python metric_train.py --generate-embeddings  # Generate & store embeddings only

After training, this script generates reference embeddings for each product
class (average of all training images) and stores them in the database.

Research Value:
    - Siamese-style training: asks "how similar?" instead of "what class?"
    - Triplet loss: (anchor, positive, negative) pulls same-class together,
      pushes different-class apart in embedding space
    - Online hard mining: picks the hardest negatives within each batch
    - Zero-shot scalability: new products added by storing ONE embedding
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import sys
import random
import argparse
from PIL import Image
from collections import defaultdict

# Add parent dir to path so we can import custom_model
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from custom_model import RetailAttnNet
from modules.vector_db import EmbeddingStore


# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset", "processed")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "retail_model.pth")
CLASS_FILE = os.path.join(BASE_DIR, "classes.txt")
DB_FILE = os.path.join(BASE_DIR, "products.db")

BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 0.0003
EMBEDDING_DIM = 256
TRIPLET_MARGIN = 0.3
# ---------------


class RandomBackground(object):
    """Pastes transparent product image onto a random solid color background."""
    def __call__(self, img):
        if img.mode == 'RGBA':
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            bg = Image.new('RGB', img.size, (r, g, b))
            bg.paste(img, mask=img.split()[3])
            return bg
        return img.convert('RGB')


class TripletDataset(Dataset):
    """
    Dataset that yields (anchor, positive, negative) triplets for metric learning.

    For each anchor image, it randomly selects:
    - A positive: another image from the SAME class
    - A hard negative: an image from a DIFFERENT class

    This enables the triplet loss to learn a meaningful embedding space.
    """

    def __init__(self, image_folder_dataset):
        self.dataset = image_folder_dataset
        self.targets = [s[1] for s in image_folder_dataset.samples]
        self.classes = image_folder_dataset.classes

        # Build index: class_id -> list of sample indices
        self.class_to_indices = defaultdict(list)
        for idx, target in enumerate(self.targets):
            self.class_to_indices[target].append(idx)

        self.class_ids = list(self.class_to_indices.keys())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # Anchor
        anchor_img, anchor_label = self.dataset[index]

        # Positive: same class, different image
        pos_indices = [i for i in self.class_to_indices[anchor_label] if i != index]
        if len(pos_indices) == 0:
            pos_indices = [index]  # Fallback: use same image
        pos_idx = random.choice(pos_indices)
        positive_img, _ = self.dataset[pos_idx]

        # Negative: different class
        neg_class = random.choice([c for c in self.class_ids if c != anchor_label])
        neg_idx = random.choice(self.class_to_indices[neg_class])
        negative_img, _ = self.dataset[neg_idx]

        return anchor_img, positive_img, negative_img, anchor_label


def get_transforms():
    """Training transforms with augmentation."""
    return transforms.Compose([
        RandomBackground(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def get_eval_transforms():
    """Evaluation transforms (no augmentation)."""
    return transforms.Compose([
        RandomBackground(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def train_metric(args):
    """Main training loop with triplet loss."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training on: {device}")

    if not os.path.exists(DATA_DIR):
        print(f"[ERROR] Dataset directory '{DATA_DIR}' not found.")
        return

    # Load dataset
    base_dataset = datasets.ImageFolder(DATA_DIR, transform=get_transforms())
    class_names = base_dataset.classes
    print(f"[INFO] Detected Classes: {class_names}")

    # Save class names
    with open(CLASS_FILE, "w") as f:
        f.write("\n".join(class_names))

    # Create triplet dataset
    triplet_dataset = TripletDataset(base_dataset)
    train_loader = DataLoader(triplet_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, drop_last=True)

    # Setup model
    model = RetailAttnNet(num_classes=len(class_names), embedding_dim=EMBEDDING_DIM)

    if args.fine_tune and os.path.exists(MODEL_SAVE_PATH):
        print("[INFO] Loading existing model for fine-tuning...")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))

    model = model.to(device)

    # Loss & Optimizer
    triplet_loss_fn = nn.TripletMarginLoss(margin=TRIPLET_MARGIN, p=2)
    # Also use classification loss as auxiliary (stabilizes training)
    cls_loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"[INFO] Training with Triplet Loss (margin={TRIPLET_MARGIN})")
    print(f"[INFO] {len(triplet_dataset)} samples, {len(class_names)} classes")

    # Training loop
    best_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        total_triplet_loss = 0.0
        total_cls_loss = 0.0
        num_batches = 0

        for anchor, positive, negative, labels in train_loader:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Get embeddings
            emb_anchor = model(anchor, return_embedding=True)
            emb_positive = model(positive, return_embedding=True)
            emb_negative = model(negative, return_embedding=True)

            # Triplet loss
            t_loss = triplet_loss_fn(emb_anchor, emb_positive, emb_negative)

            # Auxiliary classification loss (helps maintain discriminative features)
            cls_output = model(anchor, return_embedding=False)
            c_loss = cls_loss_fn(cls_output, labels)

            # Combined loss: triplet is primary, classification is auxiliary
            loss = t_loss + 0.5 * c_loss

            loss.backward()
            optimizer.step()

            total_triplet_loss += t_loss.item()
            total_cls_loss += c_loss.item()
            num_batches += 1

        scheduler.step()

        avg_t_loss = total_triplet_loss / max(num_batches, 1)
        avg_c_loss = total_cls_loss / max(num_batches, 1)

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Triplet Loss: {avg_t_loss:.4f} | "
              f"Cls Loss: {avg_c_loss:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")

        if avg_t_loss < best_loss:
            best_loss = avg_t_loss

    # Save model
    print(f"\n[INFO] Saving model to {MODEL_SAVE_PATH}...")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"[DONE] Model saved. Best Triplet Loss: {best_loss:.4f}")

    # Generate and store reference embeddings
    print("\n[INFO] Generating reference embeddings...")
    generate_reference_embeddings(model, device)


def generate_reference_embeddings(model=None, device=None):
    """
    Generate a reference embedding for each product class and store in the DB.

    For each class, takes ALL images, computes their embeddings, and averages them
    to create a robust "digital fingerprint" for that product.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model is None:
        with open(CLASS_FILE, "r") as f:
            class_names = [line.strip() for line in f.readlines()]
        model = RetailAttnNet(num_classes=len(class_names), embedding_dim=EMBEDDING_DIM)
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        model = model.to(device)

    model.eval()

    eval_transform = get_eval_transforms()
    dataset = datasets.ImageFolder(DATA_DIR, transform=eval_transform)

    # Compute embeddings per class
    class_embeddings = defaultdict(list)

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for img, label in loader:
            img = img.to(device)
            embedding = model(img, return_embedding=True)
            class_name = dataset.classes[label.item()]
            class_embeddings[class_name].append(embedding.cpu().numpy().flatten())

    # Average embeddings and store in DB
    store = EmbeddingStore(DB_FILE)

    for class_name, embeddings in class_embeddings.items():
        avg_embedding = np.mean(embeddings, axis=0)
        # Re-normalize after averaging
        avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-8)
        store.add_embedding(class_name, avg_embedding)
        print(f"  ✓ {class_name}: averaged {len(embeddings)} embeddings")

    print(f"\n[DONE] Stored reference embeddings for {len(class_embeddings)} products.") 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Metric Learning Training for ARC System")
    parser.add_argument("--fine-tune", action="store_true",
                        help="Fine-tune existing model instead of training from scratch")
    parser.add_argument("--generate-embeddings", action="store_true",
                        help="Only generate and store reference embeddings (no training)")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help=f"Number of training epochs (default: {EPOCHS})")

    args = parser.parse_args()

    if args.epochs != EPOCHS:
        EPOCHS = args.epochs

    if args.generate_embeddings:
        generate_reference_embeddings()
    else:
        train_metric(args)
