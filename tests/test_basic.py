"""Tests for multi-modal representation learning."""

import pytest
import torch
import numpy as np
from omegaconf import OmegaConf

# Add src to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.models import CLIPModel
from src.losses import ContrastiveLoss, MultiModalLoss
from src.data import ToyDataset
from src.utils import set_seed, get_device, count_parameters


class TestCLIPModel:
    """Test CLIP model functionality."""
    
    def test_model_initialization(self):
        """Test model initialization."""
        config = OmegaConf.create({
            "model_name": "openai/clip-vit-base-patch32",
            "embedding_dim": 512,
            "temperature": 0.07,
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "freeze_encoder": False,
            "fine_tune_layers": -1
        })
        
        model = CLIPModel(config)
        assert model is not None
        assert hasattr(model, "clip_model")
        assert hasattr(model, "temperature")
    
    def test_model_forward(self):
        """Test model forward pass."""
        config = OmegaConf.create({
            "model_name": "openai/clip-vit-base-patch32",
            "embedding_dim": 512,
            "temperature": 0.07,
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "freeze_encoder": False,
            "fine_tune_layers": -1
        })
        
        model = CLIPModel(config)
        
        # Create dummy inputs
        batch_size = 2
        pixel_values = torch.randn(batch_size, 3, 224, 224)
        input_ids = torch.randint(0, 1000, (batch_size, 77))
        attention_mask = torch.ones(batch_size, 77)
        
        # Forward pass
        outputs = model(pixel_values, input_ids, attention_mask)
        
        assert "image_embeddings" in outputs
        assert "text_embeddings" in outputs
        assert "logits_per_image" in outputs
        assert "logits_per_text" in outputs
        
        assert outputs["image_embeddings"].shape[0] == batch_size
        assert outputs["text_embeddings"].shape[0] == batch_size
        assert outputs["logits_per_image"].shape == (batch_size, batch_size)
        assert outputs["logits_per_text"].shape == (batch_size, batch_size)
    
    def test_model_embeddings(self):
        """Test model embedding extraction."""
        config = OmegaConf.create({
            "model_name": "openai/clip-vit-base-patch32",
            "embedding_dim": 512,
            "temperature": 0.07,
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "freeze_encoder": False,
            "fine_tune_layers": -1
        })
        
        model = CLIPModel(config)
        
        # Test image embeddings
        pixel_values = torch.randn(2, 3, 224, 224)
        image_embeddings = model.encode_image(pixel_values)
        assert image_embeddings.shape[0] == 2
        assert image_embeddings.shape[1] == config.embedding_dim
        
        # Test text embeddings
        input_ids = torch.randint(0, 1000, (2, 77))
        attention_mask = torch.ones(2, 77)
        text_embeddings = model.encode_text(input_ids, attention_mask)
        assert text_embeddings.shape[0] == 2
        assert text_embeddings.shape[1] == config.embedding_dim


class TestLossFunctions:
    """Test loss function implementations."""
    
    def test_contrastive_loss(self):
        """Test contrastive loss computation."""
        loss_fn = ContrastiveLoss(temperature=0.07)
        
        batch_size = 4
        logits_per_image = torch.randn(batch_size, batch_size)
        logits_per_text = torch.randn(batch_size, batch_size)
        
        losses = loss_fn(logits_per_image, logits_per_text)
        
        assert "contrastive_loss" in losses
        assert "loss_i2t" in losses
        assert "loss_t2i" in losses
        assert "total_loss" in losses
        
        assert losses["contrastive_loss"].item() > 0
        assert losses["loss_i2t"].item() > 0
        assert losses["loss_t2i"].item() > 0
    
    def test_multi_modal_loss(self):
        """Test multi-modal loss computation."""
        config = OmegaConf.create({
            "temperature": 0.07,
            "contrastive_loss_weight": 1.0,
            "use_triplet_loss": False,
            "use_infonce_loss": False
        })
        
        loss_fn = MultiModalLoss(config)
        
        batch_size = 4
        outputs = {
            "logits_per_image": torch.randn(batch_size, batch_size),
            "logits_per_text": torch.randn(batch_size, batch_size)
        }
        
        losses = loss_fn(outputs)
        
        assert "total_loss" in losses
        assert "contrastive_loss" in losses
        assert losses["total_loss"].item() > 0


class TestDataset:
    """Test dataset functionality."""
    
    def test_toy_dataset_initialization(self):
        """Test toy dataset initialization."""
        config = OmegaConf.create({
            "num_samples": 100,
            "train_split": 0.7,
            "val_split": 0.15,
            "test_split": 0.15,
            "image_size": 224,
            "use_augmentation": False,
            "image_mean": [0.485, 0.456, 0.406],
            "image_std": [0.229, 0.224, 0.225],
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,
            "drop_last": False
        })
        
        dataset = ToyDataset(config, split="train")
        assert len(dataset) > 0
        assert len(dataset.data) > 0
    
    def test_toy_dataset_getitem(self):
        """Test dataset item retrieval."""
        config = OmegaConf.create({
            "num_samples": 100,
            "train_split": 0.7,
            "val_split": 0.15,
            "test_split": 0.15,
            "image_size": 224,
            "use_augmentation": False,
            "image_mean": [0.485, 0.456, 0.406],
            "image_std": [0.229, 0.224, 0.225],
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,
            "drop_last": False
        })
        
        dataset = ToyDataset(config, split="train")
        
        sample = dataset[0]
        assert "image" in sample
        assert "text" in sample
        assert "category" in sample
        assert "id" in sample
        
        assert isinstance(sample["image"], torch.Tensor)
        assert isinstance(sample["text"], str)
        assert isinstance(sample["category"], str)
        assert isinstance(sample["id"], int)


class TestUtils:
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        
        # Test numpy randomness
        np.random.seed(42)
        val1 = np.random.random()
        
        set_seed(42)
        np.random.seed(42)
        val2 = np.random.random()
        
        assert val1 == val2
    
    def test_get_device(self):
        """Test device selection."""
        device = get_device("auto")
        assert isinstance(device, torch.device)
        
        device = get_device("cpu")
        assert device.type == "cpu"
    
    def test_count_parameters(self):
        """Test parameter counting."""
        config = OmegaConf.create({
            "model_name": "openai/clip-vit-base-patch32",
            "embedding_dim": 512,
            "temperature": 0.07,
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "freeze_encoder": False,
            "fine_tune_layers": -1
        })
        
        model = CLIPModel(config)
        param_counts = count_parameters(model)
        
        assert "total" in param_counts
        assert "trainable" in param_counts
        assert "frozen" in param_counts
        
        assert param_counts["total"] > 0
        assert param_counts["trainable"] > 0
        assert param_counts["frozen"] >= 0


if __name__ == "__main__":
    pytest.main([__file__])
