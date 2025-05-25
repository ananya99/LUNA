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

            # nn.Flatten(),                 # → 512 × 4 × 4 = 8192
            # nn.Linear(8192, hidden_dims)           # → Final 32-D vector
            
            nn.AdaptiveAvgPool2d((1, 1)),  # (256, 1, 1)
            nn.Flatten(),                  # → 256
            nn.Linear(256, hidden_dims)    # final 32-D vector
        )

    def forward(self, x):
        return self.encoder(x)
    
class DINOv2Encoder(nn.Module):
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
        self.model.eval()
        
        # Add an mlp layer to reduce the dimensionality to the desired hidden_dims
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(384, hidden_dims),
            nn.ReLU()
        )

    def forward(self, x):
        return self.mlp(self.model(self.transform(x)))

class ImageEncoder(nn.Module):
    def __init__(self, hidden_dims, cell_image_encoder):
        super(ImageEncoder, self).__init__()
        self.cell_image_encoder = cell_image_encoder
        
        if cell_image_encoder == "CNN":
            self.encoder = CNNEncoder(hidden_dims)
        elif cell_image_encoder == "DINOv2":
            self.encoder = DINOv2Encoder(hidden_dims)
        else:
            raise AttributeError(f"Cell image encoder {cell_image_encoder} not found")
            
    def forward(self, x):
        return self.encoder(x)

# self.encoder2 = nn.Sequential(
#     nn.Conv2d(1, 32, 3, 2, 1),  # 128 -> 64
#     nn.BatchNorm2d(32),
#     nn.ReLU(),
    
#     nn.Conv2d(32, 64, 3, 2, 1),  # 64 -> 32
#     nn.BatchNorm2d(64),
#     nn.ReLU(),

#     nn.Conv2d(64, 128, 3, 2, 1),  # 32 -> 16
#     nn.BatchNorm2d(128),
#     nn.ReLU(),

#     nn.AdaptiveAvgPool2d((1, 1)),  # spatial -> (1,1)
#     nn.Flatten(),
#     nn.Linear(128, hidden_dims["cell_image_embedding_dim"])
# )

# encoder3
# from torchvision.models.vision_transformer import vit_b_16

# vit = vit_b_16(weights=None)
# vit.conv_proj = nn.Conv2d(1, vit.conv_proj.out_channels, kernel_size=16, stride=16)
# vit.heads = nn.Linear(vit.heads.in_features, hidden_dims["cell_image_embedding_dim"])

# self.encoder = vit