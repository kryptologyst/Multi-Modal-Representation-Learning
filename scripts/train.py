#!/usr/bin/env python3
"""Training script for multi-modal representation learning."""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from omegaconf import OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data import create_data_loaders
from src.models import CLIPModel
from src.losses import MultiModalLoss
from src.eval import MultiModalEvaluator
from src.viz import MultiModalVisualizer
from src.utils import (
    set_seed, get_device, count_parameters, save_checkpoint, 
    load_checkpoint, EarlyStopping, format_time
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Trainer:
    """Trainer for multi-modal representation learning."""
    
    def __init__(self, config: OmegaConf):
        """Initialize trainer.
        
        Args:
            config: Configuration object.
        """
        self.config = config
        
        # Set random seed
        set_seed(config.seed)
        
        # Set device
        self.device = get_device(config.device)
        
        # Create output directories
        self._create_directories()
        
        # Initialize components
        self._initialize_components()
        
        # Initialize logging
        self._initialize_logging()
        
        logger.info("Trainer initialized successfully")
    
    def _create_directories(self) -> None:
        """Create necessary directories."""
        directories = [
            self.config.output_dir,
            self.config.checkpoint_dir,
            self.config.log_dir,
            self.config.asset_dir
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _initialize_components(self) -> None:
        """Initialize model, data loaders, loss, and optimizer."""
        # Create data loaders
        self.train_loader, self.val_loader, self.test_loader = create_data_loaders(
            self.config.data
        )
        
        # Initialize model
        self.model = CLIPModel(self.config.model).to(self.device)
        
        # Initialize loss function
        self.criterion = MultiModalLoss(self.config.model)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.model.learning_rate,
            weight_decay=self.config.model.weight_decay
        )
        
        # Initialize scheduler
        if self.config.train.lr_scheduler == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.train.num_epochs
            )
        else:
            self.scheduler = None
        
        # Initialize early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config.train.early_stopping_patience,
            min_delta=self.config.train.early_stopping_min_delta
        )
        
        # Initialize evaluator
        self.evaluator = MultiModalEvaluator(self.config.eval)
        
        # Initialize visualizer
        self.visualizer = MultiModalVisualizer(self.config)
        
        logger.info(f"Model parameters: {count_parameters(self.model)}")
    
    def _initialize_logging(self) -> None:
        """Initialize logging components."""
        # TensorBoard writer
        self.writer = SummaryWriter(self.config.log_dir)
        
        # Initialize wandb if enabled
        if self.config.use_wandb:
            import wandb
            wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                config=OmegaConf.to_container(self.config, resolve=True),
                name=self.config.experiment_name
            )
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            epoch: Current epoch number.
            
        Returns:
            Dictionary of training metrics.
        """
        self.model.train()
        
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.config.train.num_epochs}",
            leave=False
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            pixel_values = batch["pixel_values"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            outputs = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Compute loss
            losses = self.criterion(outputs)
            loss = losses["total_loss"]
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.train.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.train.max_grad_norm
                )
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{total_loss / (batch_idx + 1):.4f}"
            })
            
            # Log to TensorBoard
            if batch_idx % self.config.train.log_every_n_steps == 0:
                global_step = epoch * num_batches + batch_idx
                self.writer.add_scalar("train/loss", loss.item(), global_step)
                self.writer.add_scalar("train/learning_rate", self.optimizer.param_groups[0]["lr"], global_step)
        
        # Compute average loss
        avg_loss = total_loss / num_batches
        
        # Update learning rate
        if self.scheduler is not None:
            self.scheduler.step()
        
        return {"train_loss": avg_loss}
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch.
        
        Args:
            epoch: Current epoch number.
            
        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()
        
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating", leave=False):
                # Move batch to device
                pixel_values = batch["pixel_values"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Compute loss
                losses = self.criterion(outputs)
                loss = losses["total_loss"]
                
                total_loss += loss.item()
        
        # Compute average loss
        avg_loss = total_loss / num_batches
        
        # Evaluate on validation set
        val_metrics = self.evaluator.evaluate(
            self.model, self.val_loader, self.device
        )
        
        # Add loss to metrics
        val_metrics["val_loss"] = avg_loss
        
        return val_metrics
    
    def train(self) -> None:
        """Main training loop."""
        logger.info("Starting training...")
        
        best_val_loss = float("inf")
        best_metrics = {}
        
        for epoch in range(self.config.train.num_epochs):
            # Train epoch
            train_metrics = self.train_epoch(epoch)
            
            # Validate epoch
            val_metrics = self.validate_epoch(epoch)
            
            # Log metrics
            self._log_metrics(epoch, train_metrics, val_metrics)
            
            # Save checkpoint
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                best_metrics = val_metrics
                
                checkpoint_path = Path(self.config.checkpoint_dir) / "best_model.pt"
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_metrics["val_loss"],
                    val_metrics,
                    str(checkpoint_path),
                    self.config
                )
            
            # Early stopping
            if self.early_stopping(val_metrics["val_loss"], self.model):
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        logger.info("Training completed!")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        logger.info(f"Best metrics: {best_metrics}")
    
    def _log_metrics(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]) -> None:
        """Log metrics to various backends.
        
        Args:
            epoch: Current epoch number.
            train_metrics: Training metrics.
            val_metrics: Validation metrics.
        """
        # Log to console
        logger.info(f"Epoch {epoch+1}:")
        logger.info(f"  Train Loss: {train_metrics['train_loss']:.4f}")
        logger.info(f"  Val Loss: {val_metrics['val_loss']:.4f}")
        
        # Log to TensorBoard
        for metric_name, metric_value in train_metrics.items():
            self.writer.add_scalar(f"train/{metric_name}", metric_value, epoch)
        
        for metric_name, metric_value in val_metrics.items():
            self.writer.add_scalar(f"val/{metric_name}", metric_value, epoch)
        
        # Log to wandb
        if self.config.use_wandb:
            import wandb
            wandb.log({
                "epoch": epoch,
                **{f"train/{k}": v for k, v in train_metrics.items()},
                **{f"val/{k}": v for k, v in val_metrics.items()}
            })
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on test set.
        
        Returns:
            Dictionary of test metrics.
        """
        logger.info("Evaluating on test set...")
        
        # Load best model
        checkpoint_path = Path(self.config.checkpoint_dir) / "best_model.pt"
        if checkpoint_path.exists():
            load_checkpoint(str(checkpoint_path), self.model, device=self.device)
            logger.info("Loaded best model for evaluation")
        
        # Evaluate
        test_metrics = self.evaluator.evaluate(
            self.model, self.test_loader, self.device
        )
        
        logger.info("Test metrics:")
        for metric_name, metric_value in test_metrics.items():
            logger.info(f"  {metric_name}: {metric_value:.4f}")
        
        return test_metrics
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        self.writer.close()
        
        if self.config.use_wandb:
            import wandb
            wandb.finish()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train multi-modal representation learning model")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate, don't train")
    
    args = parser.parse_args()
    
    # Load configuration
    config = OmegaConf.load(args.config)
    
    # Create trainer
    trainer = Trainer(config)
    
    try:
        if args.eval_only:
            # Only evaluate
            test_metrics = trainer.evaluate()
        else:
            # Train and evaluate
            trainer.train()
            test_metrics = trainer.evaluate()
        
        logger.info("All tasks completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()
