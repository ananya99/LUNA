import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out

class ImageEncoder(nn.Module):
    def __init__(self, hidden_dims):
        super(ImageEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            # Initial conv layer
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # ResNet blocks
            ResBlock(64, 64),
            ResBlock(64, 128, stride=2),
            ResBlock(128, 256, stride=2),
            
            # Final layers
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, hidden_dims["cell_image_dimensions"]),
            nn.LayerNorm(hidden_dims["cell_image_dimensions"]),  # Added LayerNorm
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims["cell_image_dimensions"], hidden_dims["cell_image_dimensions"])
        )
        
        # Initialize weights
        for m in self.encoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        return self.encoder(x)