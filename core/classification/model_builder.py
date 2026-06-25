import torch
import torch.nn as nn
import torch.nn.functional as F

class Exoplanet1DCNN(nn.Module):
    def __init__(self, num_classes: int = 4):
        super(Exoplanet1DCNN, self).__init__()
        
        # Input tensor shape: (Batch_Size, 1, 2001)
        # Block 1: Extracts localized raw signal edges
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        
        # Block 2: Downsamples and increases feature abstraction
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        
        # Block 3: Catches deeper geometry variations (like V-shapes vs U-shapes)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm1d(64)
        
        # Pooling layer to reduce spatial dimensions drastically
        self.pool = nn.MaxPool1d(kernel_size=4, stride=4) # 2001 -> ~500 -> ~125 -> ~31
        
        # Fully Connected Classification Head
        # Flattened size calculation: 64 channels * 31 remaining spatial points
        self.fc1 = nn.Linear(64 * 31, 128)
        self.dropout = nn.Dropout(p=0.3) # Prevents overfitting on rare transit classes
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Ensure input has the channel dimension: (Batch, 1, 2001)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
            
        # Conv Blocks with Mish/ReLU activation and pooling
        x = self.pool(F.mish(self.bn1(self.conv1(x))))
        x = self.pool(F.mish(self.bn2(self.conv2(x))))
        x = self.pool(F.mish(self.bn3(self.conv3(x))))
        
        # Flatten for Dense Layers
        x = x.view(x.size(0), -1)
        
        # Fully Connected layers with Dropout
        x = F.mish(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x) # Output raw logits (CrossEntropyLoss applies Softmax internally)
        
        return x

if __name__ == "__main__":
    # Quick structural sanity check
    model = Exoplanet1DCNN()
    dummy_input = torch.randn(2, 2001) # Batch of 2 binned light curves
    output = model(dummy_input)
    print(f"Model successfully built. Output Shape: {output.shape} (Matches 4 expected categories)")