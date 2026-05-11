import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        # 1. First Convolutional Layer: 
        # Takes in 3 channels (RGB image), outputs 16 feature maps, uses a 3x3 filter
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        
        # 2. Max Pooling Layer:
        # Reduces the spatial dimensions (width and height) by half
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 3. Second Convolutional Layer:
        # Takes in 16 feature maps, outputs 32, uses a 3x3 filter
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        
        # 4. Fully Connected (Linear) Layers:
        # After two 2x2 pooling layers, a 224x224 image becomes 56x56.
        # So we have 32 feature maps of size 56x56.
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Pass data through convolution -> ReLU activation function -> pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten the 3D tensor into a 1D vector for the fully connected layers
        x = x.view(x.size(0), -1) 
        
        # Pass through fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
