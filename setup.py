#!/usr/bin/env python3
"""Setup script for multi-modal representation learning project."""

import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    """Main setup function."""
    print("Multi-Modal Representation Learning - Setup Script")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("❌ Python 3.10+ is required")
        return 1
    
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install dependencies
    print("\nInstalling dependencies...")
    if not run_command("pip install -r requirements.txt", "Installing requirements"):
        return 1
    
    # Run installation test
    print("\nRunning installation test...")
    if not run_command("python test_installation.py", "Installation test"):
        return 1
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run training: python scripts/train.py --config configs/config.yaml")
    print("2. Launch demo: streamlit run demo/streamlit_demo.py")
    print("3. Run tests: pytest tests/")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
