import torch.nn as nn
from torchvision import models

class ImageDetector(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = models.efficientnet_b0(pretrained=False)
        self.classifier = nn.Linear(1000, num_classes)
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

class TextDetector(nn.Module):
    def __init__(self, vocab_size=50000, hidden_dim=256, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        return self.classifier(hidden[-1])

class VideoDetector(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # Simple 3D CNN for video (you can replace with more complex)
        self.conv3d = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=1)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Linear(64, num_classes)
    
    def forward(self, x):
        # x shape: (batch, channels, frames, height, width)
        features = self.conv3d(x)
        features = self.pool(features)
        features = features.view(features.size(0), -1)
        return self.classifier(features)