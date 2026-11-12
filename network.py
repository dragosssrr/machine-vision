import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

WEIGHTS_DIR = Path(__file__).resolve().parent / "data"

class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2), # c0
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), # c1
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # c2
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), # c3
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # c4
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), # c5
            nn.ReLU()
        )
        self.flatten = nn.Flatten()
        self.d0 = nn.Linear(2560, 256)
        self.task5_head = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = torch.relu(self.d0(x))
        x = self.task5_head(x)
        return x

def load_weights_into_pytorch(model,weights_dir=WEIGHTS_DIR):
    # Mapping Conv layers (PyTorch indexing)
    conv_indices = [0, 2, 5, 7, 10, 12]
    
    for idx, name in zip(conv_indices, ['c0', 'c1', 'c2', 'c3', 'c4', 'c5']):
        w_path = weights_dir / f"{name}w.npy"
        b_path = weights_dir / f"{name}b.npy"
        
        w = np.load(w_path)
        b = np.load(b_path)
        
        w_pt = torch.tensor(w).permute(3, 2, 0, 1)
        b_pt = torch.tensor(b)
        
        model.features[idx].weight.data.copy_(w_pt)
        model.features[idx].bias.data.copy_(b_pt)

    # Load Dense layer d0
    d0_w = np.load(weights_dir / "d0w.npy")
    d0_b = np.load(weights_dir / "d0b.npy")
    
    model.d0.weight.data.copy_(torch.tensor(d0_w).T)
    model.d0.bias.data.copy_(torch.tensor(d0_b))

# Initialize and load
model = ConvNet(num_classes=10)
load_weights_into_pytorch(model)

# Training Setup
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Example Training Step
# outputs = model(images)
# loss = criterion(outputs, labels)
# loss.backward()
# optimizer.step()