import torch
import torch.nn as nn
from torchvision import models
from custom_model import RetailAttnNet
import os

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_size_mb(model):
    torch.save(model.state_dict(), "temp.pth")
    size = os.path.getsize("temp.pth") / (1024 * 1024)
    os.remove("temp.pth")
    return size

print(f"{'Model':<20} | {'Parameters':<15} | {'Size (MB)':<10}")
print("-" * 50)

# 1. Measure MobileNetV2
mobilenet = models.mobilenet_v2()

mobilenet.classifier[1] = nn.Linear(mobilenet.last_channel, 8) 
m_params = count_parameters(mobilenet)
m_size = get_model_size_mb(mobilenet)
print(f"{'MobileNetV2':<20} | {m_params:<15,} | {m_size:<10.2f}")

# 2. Measure Custom Model
# (Assuming 8 classes)
custom = RetailAttnNet(num_classes=8)
c_params = count_parameters(custom)
c_size = get_model_size_mb(custom)
print(f"{'RetailAttnNet':<20} | {c_params:<15,} | {c_size:<10.2f}")

print("-" * 50)
print(f"Reduction in Params: {((m_params - c_params)/m_params)*100:.1f}%")
print(f"Reduction in Size:   {((m_size - c_size)/m_size)*100:.1f}%")
