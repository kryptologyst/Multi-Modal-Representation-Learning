"""Evaluation script for multi-modal representation learning."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any

import torch
from omegaconf import OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.models import CLIPModel
from src.data import create_data_loaders
from src.eval import MultiModalEvaluator
from src.utils import get_device, load_checkpoint

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate multi-modal representation learning model")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="evaluation_results.json", help="Output file for results")
    
    args = parser.parse_args()
    
    # Load configuration
    config = OmegaConf.load(args.config)
    
    # Set device
    device = get_device(config.device)
    
    # Create data loaders
    _, _, test_loader = create_data_loaders(config.data)
    
    # Initialize model
    model = CLIPModel(config.model).to(device)
    
    # Load checkpoint
    checkpoint = load_checkpoint(args.checkpoint, model, device=device)
    logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Initialize evaluator
    evaluator = MultiModalEvaluator(config.eval)
    
    # Evaluate model
    logger.info("Starting evaluation...")
    metrics = evaluator.evaluate(model, test_loader, device)
    
    # Print results
    logger.info("Evaluation Results:")
    for metric_name, metric_value in metrics.items():
        logger.info(f"  {metric_name}: {metric_value:.4f}")
    
    # Save results
    import json
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
