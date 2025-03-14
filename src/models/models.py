"""
Model definitions for FL-for-DR.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Optional


class SimpleCNN(nn.Module):
    """
    Simple CNN model for image classification.
    """
    
    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int = 10):
        """
        Initialize the CNN model.
        
        Args:
            input_shape: Input shape (channels, height, width)
            num_classes: Number of output classes
        """
        super(SimpleCNN, self).__init__()
        
        channels, height, width = input_shape
        
        # Feature extraction layers
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Calculate size after convolutions and pooling
        conv_height = height // 4  # After 2 pooling layers
        conv_width = width // 4
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * conv_height * conv_width, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Feature extraction
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class MLP(nn.Module):
    """
    Multi-layer perceptron model.
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        hidden_dims: List[int] = [128, 64],
        num_classes: int = 10
    ):
        """
        Initialize the MLP model.
        
        Args:
            input_shape: Input shape (channels, height, width)
            hidden_dims: Dimensions of hidden layers
            num_classes: Number of output classes
        """
        super(MLP, self).__init__()
        
        channels, height, width = input_shape
        input_dim = channels * height * width
        
        # Create layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Flatten input
        x = x.view(x.size(0), -1)
        
        # Forward through layers
        x = self.layers(x)
        
        return x


class ResidualBlock(nn.Module):
    """
    Residual block for ResNet.
    """
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        """
        Initialize the residual block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Stride for the first convolution
        """
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        
        return out


class ResNet18(nn.Module):
    """
    ResNet-18 model.
    """
    
    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int = 10):
        """
        Initialize the ResNet-18 model.
        
        Args:
            input_shape: Input shape (channels, height, width)
            num_classes: Number of output classes
        """
        super(ResNet18, self).__init__()
        
        channels, _, _ = input_shape
        
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Create layers
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        """
        Create a layer of residual blocks.
        
        Args:
            out_channels: Number of output channels
            num_blocks: Number of residual blocks
            stride: Stride for the first block
            
        Returns:
            Sequential layer
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        
        for stride in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        out = F.relu(self.bn1(self.conv1(x)))
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        
        return out


def get_model(
    model_name: str,
    input_shape: Tuple[int, int, int],
    num_classes: int = 10,
    hidden_dims: Optional[List[int]] = None
) -> nn.Module:
    """
    Get a model by name.
    
    Args:
        model_name: Name of the model (e.g., cnn, mlp, resnet18)
        input_shape: Input shape (channels, height, width)
        num_classes: Number of output classes
        hidden_dims: Dimensions of hidden layers (for MLP)
        
    Returns:
        Model instance
    """
    if model_name.lower() == "cnn":
        return SimpleCNN(input_shape, num_classes)
    elif model_name.lower() == "mlp":
        if hidden_dims is None:
            hidden_dims = [128, 64]
        return MLP(input_shape, hidden_dims, num_classes)
    elif model_name.lower() == "resnet18":
        return ResNet18(input_shape, num_classes)
    else:
        raise ValueError(f"Model {model_name} not supported") 