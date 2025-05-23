import torch.nn as nn
import torch
import torchvision.transforms as T
from PIL import Image

class ImageEncoder(nn.Module):
    def __init__(self, hidden_dims, cell_image_encoder):
        super(ImageEncoder, self).__init__()
        self.cell_image_encoder = cell_image_encoder
        
        if cell_image_encoder == "cnn":
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 64, 4, 2, 1),  # downsample
                nn.ReLU(),
                nn.Conv2d(64, 128, 4, 2, 1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(128*32*32, hidden_dims)  # final latent dim
            )
        elif cell_image_encoder == "DINOv2":
            # Load model from torch.hub
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')  # You can also try dinov2_vitl14
            self.model.eval()
            transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.5]*3, std=[0.5]*3)])
            self.encoder = lambda x: self.model(transform(x))
            
    def forward(self, x):
        if self.cell_image_encoder == "cnn":
            return self.encoder(x)
        elif self.cell_image_encoder == "DINOv2":
            with torch.no_grad():
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
#     nn.Linear(128, hidden_dims["cell_image_dimensions"])
# )

# encoder3
# from torchvision.models.vision_transformer import vit_b_16

# vit = vit_b_16(weights=None)
# vit.conv_proj = nn.Conv2d(1, vit.conv_proj.out_channels, kernel_size=16, stride=16)
# vit.heads = nn.Linear(vit.heads.in_features, hidden_dims["cell_image_dimensions"])

# self.encoder = vit