import unittest
import torch
import numpy as np
import sys
import os

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model import Model
from utils.data.dataholder import DataHolder

class TestModel(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        # Define model dimensions
        self.input_dims = {
            "node_features_dimensions": 64,
            "diffusion_time_dimensions": 32
        }
        self.output_dims = {
            "node_features_dimensions": 32,
            "diffusion_time_dimensions": 16
        }
        self.hidden_mlp_dims = {
            "X": 128,
            "y": 64,
            "pos": 32
        }
        self.hidden_dims = {
            "dx": 256,
            "dy": 128,
            "dd": 64,
            "num_heads": 8,
            "dim_ffX": 512,
            "dim_ffy": 256,
            "cell_image_embedding_dim": 32,
            "output_features_to_pos_dims": 4
        }
        
        # Initialize model
        self.model = Model(
            input_dims=self.input_dims,
            n_layers=2,
            hidden_mlp_dims=self.hidden_mlp_dims,
            hidden_dims=self.hidden_dims,
            output_dims=self.output_dims
        )
        
        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def test_model_initialization(self):
        """Test model initialization."""
        self.assertIsInstance(self.model, Model)
        self.assertEqual(self.model.n_layers, 2)
        self.assertEqual(self.model.input_dimensions_node_features, 64)
        self.assertEqual(self.model.input_dimensions_diffusion_time, 32)

    def test_forward_pass_without_images(self):
        """Test forward pass without cell images."""
        # Create test data
        batch_size = 4
        num_nodes = 10
        
        # Create random input tensors
        node_features = torch.randn(batch_size, num_nodes, self.input_dims["node_features_dimensions"])
        diffusion_time = torch.randn(batch_size, self.input_dims["diffusion_time_dimensions"])
        positions = torch.randn(batch_size, num_nodes, 2)  # 2D positions (x, y)
        node_mask = torch.ones(batch_size, num_nodes, dtype=torch.bool)
        
        # Move to device
        node_features = node_features.to(self.device)
        diffusion_time = diffusion_time.to(self.device)
        positions = positions.to(self.device)
        node_mask = node_mask.to(self.device)
        
        # Create DataHolder
        data = DataHolder(
            node_features=node_features,
            diffusion_time=diffusion_time,
            positions=positions,
            node_mask=node_mask
        )
        
        # Forward pass
        output = self.model(data)
        
        # Check output shapes
        # self.assertEqual(output.node_features.shape, (batch_size, num_nodes, self.output_dims["node_features_dimensions"]))
        self.assertEqual(output.diffusion_time.shape, (batch_size, self.output_dims["diffusion_time_dimensions"]))
        self.assertEqual(output.positions.shape, (batch_size, num_nodes, 2))
        self.assertEqual(output.node_mask.shape, (batch_size, num_nodes))

    def test_forward_pass_with_images(self):
        """Test forward pass with cell images."""
        # Create test data
        batch_size = 4
        num_nodes = 10
        
        # Create random input tensors
        node_features = torch.randn(batch_size, num_nodes, self.input_dims["node_features_dimensions"])
        diffusion_time = torch.randn(batch_size, self.input_dims["diffusion_time_dimensions"])
        positions = torch.randn(batch_size, num_nodes, 2)  # 2D positions (x, y)
        node_mask = torch.ones(batch_size, num_nodes, dtype=torch.bool)
        # Use raw cell images (128x128) instead of encoded ones
        cell_images = torch.randn(batch_size, num_nodes, 128, 128)  # Raw grayscale images
        
        # Move to device
        node_features = node_features.to(self.device)
        diffusion_time = diffusion_time.to(self.device)
        positions = positions.to(self.device)
        node_mask = node_mask.to(self.device)
        cell_images = cell_images.to(self.device)
        
        # Create DataHolder
        data = DataHolder(
            node_features=node_features,
            cell_images=cell_images,
            diffusion_time=diffusion_time,
            positions=positions,
            node_mask=node_mask
        )
        
        # Forward pass
        output = self.model(data)
        
        # Check output shapes
        # self.assertEqual(output.node_features.shape, (batch_size, num_nodes, self.output_dims["node_features_dimensions"]))
        self.assertEqual(output.diffusion_time.shape, (batch_size, self.output_dims["diffusion_time_dimensions"]))
        self.assertEqual(output.positions.shape, (batch_size, num_nodes, 2))
        self.assertEqual(output.node_mask.shape, (batch_size, num_nodes))


    # def test_gradient_flow(self):
    #     """Test gradient flow through the model."""
    #     # Create test data
    #     batch_size = 4
    #     num_nodes = 10
        
    #     # Create random input tensors
    #     node_features = torch.randn(batch_size, num_nodes, self.input_dims["node_features_dimensions"])
    #     diffusion_time = torch.randn(batch_size, num_nodes, self.input_dims["diffusion_time_dimensions"])
    #     positions = torch.randn(batch_size, num_nodes, 3)
    #     node_mask = torch.ones(batch_size, num_nodes, dtype=torch.bool)
    #     cell_images = torch.randn(batch_size, num_nodes, 128, 128)
        
    #     # Move to device
    #     node_features = node_features.to(self.device)
    #     diffusion_time = diffusion_time.to(self.device)
    #     positions = positions.to(self.device)
    #     node_mask = node_mask.to(self.device)
    #     cell_images = cell_images.to(self.device)
        
    #     # Create DataHolder
    #     data = DataHolder(
    #         node_features=node_features,
    #         cell_images=cell_images,
    #         diffusion_time=diffusion_time,
    #         positions=positions,
    #         node_mask=node_mask
    #     )
        
    #     # Forward pass
    #     output = self.model(data)
        
    #     # Compute loss and backward pass
    #     loss = output.node_features.sum() + output.positions.sum()
    #     loss.backward()
        
    #     # Check gradients
    #     for name, param in self.model.named_parameters():
    #         self.assertIsNotNone(param.grad, f"Gradient is None for parameter {name}")
    #         self.assertFalse(torch.isnan(param.grad).any(), f"Gradient contains NaN for parameter {name}")

if __name__ == '__main__':
    unittest.main() 