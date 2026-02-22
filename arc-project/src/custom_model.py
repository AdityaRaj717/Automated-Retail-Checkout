import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block.
    Research Value: Allows the network to adaptively recalibrate channel-wise feature responses.
    It models interdependencies between channels (e.g., 'is this red color a label or a fruit?').
    """
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class RetailBlock(nn.Module):
    """
    A custom convolutional block combining Conv, BatchNorm, and Attention.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(RetailBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)  # The Research Component

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out) # Apply Attention
        out += self.shortcut(x) # Residual Connection (ResNet style)
        out = F.relu(out)
        return out

class RetailAttnNet(nn.Module):
    """
    Custom ResNet-style backbone with SE attention blocks.

    Supports two modes:
        1. Classification mode (default): Returns class logits.
        2. Embedding mode (return_embedding=True): Returns L2-normalized
           256-dim vectors for metric learning / few-shot recognition.

    Args:
        num_classes: Number of classification categories.
        embedding_dim: Dimension of the embedding vector (default: 256).
    """
    def __init__(self, num_classes=10, embedding_dim=256):
        super(RetailAttnNet, self).__init__()
        self.in_channels = 64
        self.embedding_dim = embedding_dim

        # Initial Feature Extraction
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Custom Layers (The "Brain")
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)

        # Global pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Embedding Head — projects features to a compact vector space
        self.embedding_head = nn.Sequential(
            nn.Linear(256, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

        # Classifier Head — sits on top of embeddings for classification mode
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def _make_layer(self, out_channels, blocks, stride):
        layers = []
        layers.append(RetailBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(RetailBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x, return_embedding=False):
        """
        Forward pass with dual output mode.

        Args:
            x: Input tensor of shape (B, 3, H, W)
            return_embedding: If True, returns L2-normalized embedding vector.
                              If False, returns classification logits.

        Returns:
            If return_embedding: (B, embedding_dim) L2-normalized embeddings
            Else: (B, num_classes) classification logits
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # Compute embedding
        embedding = self.embedding_head(x)
        embedding = F.normalize(embedding, p=2, dim=1)  # L2 normalize to unit sphere

        if return_embedding:
            return embedding

        # Classification mode
        return self.classifier(embedding)
