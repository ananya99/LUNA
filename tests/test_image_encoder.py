import os
import sys
import pytest
import torch
import torch.nn as nn

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.image_encoder import ImageEncoder

@pytest.fixture
def sample_input():
    # Create a batch of 2 grayscale images of size 128x128
    return torch.randn(2, 1, 128, 128)

@pytest.fixture
def hidden_dims():
    return 256

def test_cnn_encoder_initialization(hidden_dims):
    encoder = ImageEncoder(hidden_dims=hidden_dims, cell_image_encoder="cnn")
    assert isinstance(encoder, nn.Module)
    assert encoder.cell_image_encoder == "cnn"
    assert isinstance(encoder.encoder, nn.Sequential)

def test_dinov2_encoder_initialization(hidden_dims):
    encoder = ImageEncoder(hidden_dims=hidden_dims, cell_image_encoder="DINOv2")
    assert isinstance(encoder, nn.Module)
    assert encoder.cell_image_encoder == "DINOv2"
    assert hasattr(encoder, 'model')
    assert hasattr(encoder, 'transform')

def test_cnn_encoder_forward(sample_input, hidden_dims):
    encoder = ImageEncoder(hidden_dims=hidden_dims, cell_image_encoder="cnn")
    output = encoder(sample_input)
    
    # Check output shape
    assert output.shape == (2, hidden_dims)  # batch_size x hidden_dims
    assert isinstance(output, torch.Tensor)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()

def test_dinov2_encoder_forward(sample_input, hidden_dims):
    encoder = ImageEncoder(hidden_dims=hidden_dims, cell_image_encoder="DINOv2")
    output = encoder(sample_input)
    
    # Check output shape and properties
    assert isinstance(output, torch.Tensor)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()
    # DINOv2 outputs 384-dimensional features by default
    assert output.shape[1] == 384

def test_invalid_encoder_type():
    with pytest.raises(AttributeError):
        ImageEncoder(hidden_dims=256, cell_image_encoder="invalid_type")

def test_cnn_encoder_gradient_flow(sample_input, hidden_dims):
    encoder = ImageEncoder(hidden_dims=hidden_dims, cell_image_encoder="cnn")
    output = encoder(sample_input)
    
    # Test gradient flow
    loss = output.mean()
    loss.backward()
    
    # Check if gradients are computed
    for param in encoder.parameters():
        assert param.grad is not None
        assert not torch.isnan(param.grad).any()
        assert not torch.isinf(param.grad).any()

def test_dinov2_encoder_no_gradients(sample_input, hidden_dims):
    encoder = ImageEncoder(hidden_dims=hidden_dims, cell_image_encoder="DINOv2")
    output = encoder(sample_input)
    
    # DINOv2 should be in eval mode and not compute gradients
    assert encoder.model.training == False
    assert not output.requires_grad

if __name__ == "__main__":
    pytest.main()
