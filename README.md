# Retail Checkout System (Phase 2)

**Automated, AI-Powered Retail Billing via Metric Learning**

This repository contains the complete Phase 2 source code for an automated retail checkout system designed to replace traditional barcode scanning. A camera captures the checkout table, and our custom computer vision pipeline identifies all products, calculates their prices, and generates a bill—all without manual scanning.

### The Core Innovation: Zero-Retraining Pipeline
Standard classifiers (YOLO, end-to-end ResNet) require extensive retraining whenever a new product is added to a store's inventory. We solved this scalability bottleneck by implementing a **Metric Learning Pipeline**:
1. Products are converted into **512-dimensional vector embeddings** using a headless ResNet18 feature extractor.
2. Classification happens instantly via **k-Nearest Neighbors (kNN)** similarity search against a vector database.
3. Adding a new product requires **zero model retraining** (fit time: `0.48 ms` vs YOLO's `~5 mins`).

---

## Technical Architecture

The system utilizes a decoupled, asynchronous multi-stage pipeline designed for **30fps real-time inference**:

| Component | Model / Algorithm | Purpose |
|-----------|------------------|---------|
| **Segmentation** | ISNet / Dichotomous Image Segmentation (DIS) | Pixel-level foreground extraction based on U^2-Net topology to isolate products. |
| **Feature Extraction** | ResNet18 (ImageNet Pre-trained) | Acts as a Digital Fingerprinter to output 512-D embeddings instead of classification logits. |
| **Classification** | k-Nearest Neighbors (k=5, cosine) | Matches live embeddings against stored catalog using vector similarity. |
| **Volumetric Depth** | Depth Anything V2 + Custom HBAO | Generates monocular depth and Horizon-Based Ambient Occlusion maps to differentiate size variants (e.g., 10rs vs 30rs packets) and resolve stacking occlusion. |
| **Blob Separation** | Distance-Transform Watershed | Mathematically splits products that touch physically in the segmentation mask. |

### The Detection Flow
For full documentation on the pipeline workflow (including BBox cropping, IoU deduplication, sliding window fallbacks, and temporal consensus smoothing), please refer to the comprehensive [**Project Guide**](arc-project/docs/PROJECT_GUIDE.md).

---

## Quick Start Guide

### 1. Start the FastAPI Backend
The backend handles the camera feed, object detection, embedding similarity search, and SQLite transactions.
```bash
cd arc-project
source .venv/bin/activate
python server.py
# API runs at http://localhost:8000
```

### 2. Start the Next.js Frontend Dashboard
The React-based dashboard features a 3-column layout: live DroidCam feed, detection analytics/vision maps, and dynamic bill management.
```bash
cd arc-project/web
npm run dev
# Dashboard runs at http://localhost:3000
```

### 3. Adding New Products (Zero-Retraining)
If you add new images to `dataset/processed/<new_product>`, you just need to rebuild the vector database:
```bash
cd arc-project
source .venv/bin/activate
python build_embeddings.py
```
Then click "Reload Embeddings" on the web dashboard to hot-reload the changes without restarting the server.

---

## Documentation & Presentation

- **Comprehensive Technical Guide:** `arc-project/docs/PROJECT_GUIDE.md`
- **Supervisor Presentation:** Generated via `arc-project/presentation/ppt_gen_v2.js`
- **Literature Survey & Citations:** Generated via `arc-project/docx/generate_literature_survey.js`

> *Built for Capstone Phase 2 | Group 11 | VIT Bhopal*
