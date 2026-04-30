#!/usr/bin/env python3
"""Simple test script to verify the installation and basic functionality."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from src.models import CLIPModel
        print("✓ CLIPModel imported successfully")
    except Exception as e:
        print(f"✗ Failed to import CLIPModel: {e}")
        return False
    
    try:
        from src.data import ToyDataset
        print("✓ ToyDataset imported successfully")
    except Exception as e:
        print(f"✗ Failed to import ToyDataset: {e}")
        return False
    
    try:
        from src.losses import ContrastiveLoss
        print("✓ ContrastiveLoss imported successfully")
    except Exception as e:
        print(f"✗ Failed to import ContrastiveLoss: {e}")
        return False
    
    try:
        from src.eval import MultiModalEvaluator
        print("✓ MultiModalEvaluator imported successfully")
    except Exception as e:
        print(f"✗ Failed to import MultiModalEvaluator: {e}")
        return False
    
    try:
        from src.utils import set_seed, get_device
        print("✓ Utils imported successfully")
    except Exception as e:
        print(f"✗ Failed to import utils: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality."""
    print("\nTesting basic functionality...")
    
    try:
        from src.utils import set_seed, get_device
        from src.models import CLIPModel
        from omegaconf import OmegaConf
        
        # Set seed
        set_seed(42)
        print("✓ Seed set successfully")
        
        # Get device
        device = get_device("cpu")
        print(f"✓ Device: {device}")
        
        # Create model config
        config = OmegaConf.create({
            "model_name": "openai/clip-vit-base-patch32",
            "embedding_dim": 512,
            "temperature": 0.07,
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "freeze_encoder": False,
            "fine_tune_layers": -1
        })
        
        # Initialize model
        model = CLIPModel(config)
        print("✓ Model initialized successfully")
        
        # Test parameter counting
        param_counts = model.count_parameters()
        print(f"✓ Model parameters: {param_counts['total']:,}")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def main():
    """Main test function."""
    print("Multi-Modal Representation Learning - Installation Test")
    print("=" * 60)
    
    # Test imports
    imports_ok = test_imports()
    
    if imports_ok:
        # Test basic functionality
        functionality_ok = test_basic_functionality()
        
        if functionality_ok:
            print("\n🎉 All tests passed! Installation is working correctly.")
            print("\nNext steps:")
            print("1. Run training: python scripts/train.py --config configs/config.yaml")
            print("2. Launch demo: streamlit run demo/streamlit_demo.py")
            return 0
        else:
            print("\n❌ Basic functionality test failed.")
            return 1
    else:
        print("\n❌ Import test failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
