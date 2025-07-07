import torch.nn as nn
import torch
import torchvision.transforms as T
from PIL import Image

class GrayscaleToRGB(nn.Module):
    def forward(self, x):
        return x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x

class Normalize(nn.Module):
    def forward(self, x):
        return (x - 0.5) / 0.5  # Normalize to [-1, 1] range
    
class CNNEncoder(nn.Module):
    """
    Basic CNN encoder for grayscale cell images.
    
    Architecture:
        - 4x4 Conv (stride=2, padding=1), 32 filters → GroupNorm → ReLU
        - 4x4 Conv (stride=2, padding=1), 64 filters → GroupNorm → ReLU
        - 4x4 Conv (stride=2, padding=1), 128 filters → GroupNorm → ReLU
        - 4x4 Conv (stride=2, padding=1), 256 filters → GroupNorm → ReLU
        - AdaptiveAvgPool2d((1, 1)) → Flatten → Linear(256 → hidden_dims)
        
    Args:
        hidden_dims (int): Size of hidden layer (default: 32)
        
    Forward:
        Input: 1-channel grayscale image (e.g., 128x128)
        Output: hidden_dims-dimensional vector (e.g., 32-D)
    """
    def __init__(self, hidden_dims):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),     # → (32, 64, 64)
            nn.GroupNorm(4, 32),
            nn.ReLU(),

            nn.Conv2d(32, 64, 4, 2, 1),   # → (64, 32, 32)
            nn.GroupNorm(4, 64),
            nn.ReLU(),

            nn.Conv2d(64, 128, 4, 2, 1),  # → (128, 16, 16)
            nn.GroupNorm(8, 128),
            nn.ReLU(),

            nn.Conv2d(128, 256, 4, 2, 1),  # → (256, 8, 8)
            nn.GroupNorm(16, 256),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool2d((1, 1)),  # → (256, 1, 1)
            nn.Flatten(),                  # → 256
            nn.Linear(256, hidden_dims)    # → final 32-D vector
        )

    def forward(self, x):
        return self.encoder(x)
    
class DINOv2Encoder(nn.Module):
    """
    DINOv2-based encoder for grayscale cell images.

    - Converts grayscale images to 3-channel RGB
    - Resizes and normalizes input to match DINOv2 expectations
    - Uses pretrained DINOv2 ViT-S/14 backbone (frozen)
    - Outputs a `hidden_dims`-dimensional feature vector via a linear MLP head

    Args:
        hidden_dims (int): Size of hidden layer (default: 32)
        
    Forward:
        Input: 1xHxW grayscale image
        Output: hidden_dims-dimensional vector (e.g., 32-D)
    """
    def __init__(self, hidden_dims):
        super().__init__()

        # Define transforms for grayscale to RGB conversion and preprocessing
        self.transform = nn.Sequential(
            # Convert grayscale to RGB by repeating the channel
            GrayscaleToRGB(),
            # Resize to DINOv2's expected input size
            nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False),
            # Normalize after resize
            Normalize()
        )
        
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        
        # Freeze DINOv2 weights
        for param in self.model.parameters():
            param.requires_grad = False  # freeze DINOv2 weights
        self.model.eval()
        
        # Add an mlp layer to reduce the dimensionality to the desired hidden_dims
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(384, hidden_dims)
        )

    def forward(self, x):
        x = self.transform(x)
        with torch.no_grad():  # don't compute gradients through DINOv2
            x = self.model(x)
        return self.mlp(x)

class ImageEncoder(nn.Module):
    """
    Image encoder for cell images. It can be a CNN, DINOv2 or MAE embeddings encoder based on the cell_image_encoder argument.

    Args:
        output_dims (int): Size of output feature vector (default: 32)
        cell_image_encoder (str): Type of cell image encoder (default: None). Options: "CNN", "DINOv2", "MAE_embeddings"
    """
    def __init__(self, output_dims, cell_image_encoder):
        super(ImageEncoder, self).__init__()
        self.cell_image_encoder = cell_image_encoder
        
        if cell_image_encoder is None:
            print("No cell image encoder")
            pass
        elif cell_image_encoder == "CNN":
            self.encoder = CNNEncoder(output_dims)
        elif cell_image_encoder == "DINOv2":
            self.encoder = DINOv2Encoder(output_dims)
        elif cell_image_encoder == "MAE_embeddings":
            self.encoder = MAEEmbeddingsMLP(output_dims=output_dims)
        else:
            raise AttributeError(f"Invalid cell image encoder: {cell_image_encoder}")
            
    def forward(self, x):
        return self.encoder(x)
    
class MAEEmbeddingsMLP(nn.Module):
    """
    MLP for reducing MAE embedding vectors.

    Args:
        input_dims (int): Dimensionality of input embeddings (default: 768)
        hidden_dims (int): Size of hidden layer (default: 128)
        output_dims (int): Size of output feature vector (default: 32)

    Forward:
        Input: Tensor of shape [B, input_dims]
        Output: Tensor of shape [B, output_dims]
    """
    def __init__(self, input_dims = 768, hidden_dims = 128, output_dims = 32):
        super(MAEEmbeddingsMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, output_dims),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.mlp(x)