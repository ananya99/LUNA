import torch
import torch.nn as nn

class FusionModule(nn.Module):
    def __init__(self, hidden_dims, fusion_type="concat"):
        super(FusionModule, self).__init__()
        self.fusion_type = fusion_type
        
        # Dimensions
        self.gene_dim = hidden_dims["dx"]  # gene expression dimension
        self.image_dim = hidden_dims["cell_image_dimensions"]  # image feature dimension
        self.output_dim = hidden_dims["dx"]  # output dimension (same as gene dim)
        
        if fusion_type == "concat":
            # Simple concatenation followed by MLP
            self.fusion_layer = nn.Sequential(
                nn.Linear(self.gene_dim + self.image_dim, self.output_dim),
                nn.LayerNorm(self.output_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            )
            
        elif fusion_type == "attention":
            # Cross-attention between modalities
            self.query_proj = nn.Linear(self.gene_dim, self.output_dim)
            self.key_proj = nn.Linear(self.image_dim, self.output_dim)
            self.value_proj = nn.Linear(self.image_dim, self.output_dim)
            self.attention = nn.MultiheadAttention(self.output_dim, num_heads=4, batch_first=True)
            self.norm = nn.LayerNorm(self.output_dim)
            
        elif fusion_type == "gated":
            # Gated fusion mechanism
            self.gate = nn.Sequential(
                nn.Linear(self.gene_dim + self.image_dim, self.output_dim),
                nn.Sigmoid()
            )
            self.transform = nn.Sequential(
                nn.Linear(self.gene_dim + self.image_dim, self.output_dim),
                nn.LayerNorm(self.output_dim),
                nn.ReLU(inplace=True)
            )
    
    def forward(self, gene_features, image_features):
        """
        Args:
            gene_features: [batch_size, num_cells, gene_dim]
            image_features: [batch_size, num_cells, image_dim]
        Returns:
            fused_features: [batch_size, num_cells, output_dim]
        """
        if self.fusion_type == "concat":
            # Simple concatenation
            combined = torch.cat([gene_features, image_features], dim=-1)
            return self.fusion_layer(combined)
            
        elif self.fusion_type == "attention":
            # Cross-attention: gene features attend to image features
            query = self.query_proj(gene_features)
            key = self.key_proj(image_features)
            value = self.value_proj(image_features)
            
            attended_features, _ = self.attention(query, key, value)
            fused = gene_features + attended_features  # residual connection
            return self.norm(fused)
            
        elif self.fusion_type == "gated":
            # Gated fusion
            combined = torch.cat([gene_features, image_features], dim=-1)
            gate_values = self.gate(combined)
            transformed = self.transform(combined)
            return gate_values * transformed + (1 - gate_values) * gene_features  # gated skip connection 