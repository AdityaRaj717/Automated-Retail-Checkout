"""
Build Embeddings Pipeline
=========================
Loads all background-removed product images from dataset/processed/<product>/,
extracts feature embeddings using a pre-trained ResNet18, and saves them to
embeddings.pkl for use in real-time classification.

Usage:
    python build_embeddings.py
"""

import os
import pickle
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

# ── Config ──────────────────────────────────────────────────────────────────
DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset", "processed")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "embeddings.pkl")

# ── Model Setup ─────────────────────────────────────────────────────────────
def get_feature_extractor():
    """Load ResNet18 and remove the final classification layer to get a 512-d
    feature extractor."""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Remove the final FC layer → output is 512-d feature vector
    model = nn.Sequential(*list(model.children())[:-1])
    model.eval()
    return model


# Standard ImageNet preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_and_preprocess(image_path: str) -> torch.Tensor:
    """Load an RGBA image, composite onto white background, and preprocess."""
    img = Image.open(image_path).convert("RGBA")

    # Composite onto white background (since rembg gives transparent BG)
    background = Image.new("RGBA", img.size, (255, 255, 255, 255))
    composite = Image.alpha_composite(background, img).convert("RGB")

    return preprocess(composite)


def extract_embeddings():
    """Walk through the processed dataset and extract embeddings for every
    product image."""
    model = get_feature_extractor()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    embeddings = []
    labels = []

    product_dirs = sorted([
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    ])

    print(f"Found {len(product_dirs)} product categories")
    print(f"Using device: {device}")
    print()

    for product_name in product_dirs:
        product_path = os.path.join(DATASET_DIR, product_name)
        image_files = sorted([
            f for f in os.listdir(product_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

        print(f"  {product_name}: {len(image_files)} images ... ", end="", flush=True)

        batch_tensors = []
        valid_count = 0

        for img_file in image_files:
            img_path = os.path.join(product_path, img_file)
            try:
                tensor = load_and_preprocess(img_path)
                batch_tensors.append(tensor)
                labels.append(product_name)
                valid_count += 1
            except Exception as e:
                print(f"\n    [WARN] Skipping {img_file}: {e}")

        if batch_tensors:
            # Process as a batch for efficiency
            batch = torch.stack(batch_tensors).to(device)
            with torch.no_grad():
                features = model(batch)  # (N, 512, 1, 1)
                features = features.squeeze(-1).squeeze(-1)  # (N, 512)
                # L2 normalize for better cosine similarity
                features = torch.nn.functional.normalize(features, p=2, dim=1)
                embeddings.append(features.cpu().numpy())

        print(f"✓ ({valid_count} embedded)")

    # Stack all embeddings into a single array
    all_embeddings = np.concatenate(embeddings, axis=0)

    data = {
        "embeddings": all_embeddings,
        "labels": labels,
        "product_names": product_dirs,
    }

    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(data, f)

    print(f"\n{'='*50}")
    print(f"Saved {all_embeddings.shape[0]} embeddings ({all_embeddings.shape[1]}-d)")
    print(f"Product classes: {product_dirs}")
    print(f"Output: {OUTPUT_FILE}")

def build():
    """Build embeddings — callable from server or CLI."""
    extract_embeddings()


if __name__ == "__main__":
    build()
