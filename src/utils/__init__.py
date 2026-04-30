"""Utility functions for multi-modal representation learning."""

import random
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from omegaconf import DictConfig

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables for additional reproducibility
    import os
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    logger.info(f"Random seed set to {seed}")


def get_device(device: str = "auto") -> torch.device:
    """Get the appropriate device for computation.
    
    Args:
        device: Device specification ("auto", "cuda", "mps", "cpu").
        
    Returns:
        PyTorch device object.
    """
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    device_obj = torch.device(device)
    logger.info(f"Using device: {device_obj}")
    return device_obj


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count the number of parameters in a model.
    
    Args:
        model: PyTorch model.
        
    Returns:
        Dictionary with total and trainable parameter counts.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total": total_params,
        "trainable": trainable_params,
        "frozen": total_params - trainable_params
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    metrics: Dict[str, float],
    filepath: str,
    config: Optional[DictConfig] = None
) -> None:
    """Save model checkpoint.
    
    Args:
        model: PyTorch model to save.
        optimizer: Optimizer state.
        epoch: Current epoch number.
        loss: Current loss value.
        metrics: Dictionary of metrics.
        filepath: Path to save checkpoint.
        config: Configuration object.
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "metrics": metrics,
        "config": config
    }
    
    torch.save(checkpoint, filepath)
    logger.info(f"Checkpoint saved to {filepath}")


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint file.
        model: PyTorch model to load state into.
        optimizer: Optional optimizer to load state into.
        device: Device to load checkpoint on.
        
    Returns:
        Dictionary containing checkpoint information.
    """
    if device is None:
        device = next(model.parameters()).device
    
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    logger.info(f"Checkpoint loaded from {filepath}")
    return checkpoint


def create_attention_visualization(
    attention_weights: torch.Tensor,
    image: Image.Image,
    text_tokens: List[str],
    save_path: Optional[str] = None
) -> plt.Figure:
    """Create attention visualization for image-text pairs.
    
    Args:
        attention_weights: Attention weights tensor.
        image: PIL Image object.
        text_tokens: List of text tokens.
        save_path: Optional path to save the figure.
        
    Returns:
        Matplotlib figure object.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Display image
    ax1.imshow(image)
    ax1.set_title("Input Image")
    ax1.axis("off")
    
    # Display attention weights
    attention_np = attention_weights.detach().cpu().numpy()
    sns.heatmap(
        attention_np,
        xticklabels=text_tokens,
        yticklabels=False,
        cmap="Blues",
        ax=ax2
    )
    ax2.set_title("Attention Weights")
    ax2.set_xlabel("Text Tokens")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Attention visualization saved to {save_path}")
    
    return fig


def compute_similarity_matrix(
    embeddings1: torch.Tensor,
    embeddings2: torch.Tensor,
    normalize: bool = True
) -> torch.Tensor:
    """Compute similarity matrix between two sets of embeddings.
    
    Args:
        embeddings1: First set of embeddings.
        embeddings2: Second set of embeddings.
        normalize: Whether to normalize embeddings before computing similarity.
        
    Returns:
        Similarity matrix.
    """
    if normalize:
        embeddings1 = torch.nn.functional.normalize(embeddings1, p=2, dim=-1)
        embeddings2 = torch.nn.functional.normalize(embeddings2, p=2, dim=-1)
    
    similarity_matrix = torch.matmul(embeddings1, embeddings2.T)
    return similarity_matrix


def compute_retrieval_metrics(
    similarity_matrix: torch.Tensor,
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """Compute retrieval metrics from similarity matrix.
    
    Args:
        similarity_matrix: Similarity matrix between queries and candidates.
        k_values: List of k values for recall@k computation.
        
    Returns:
        Dictionary of retrieval metrics.
    """
    # Get the diagonal indices (correct matches)
    batch_size = similarity_matrix.size(0)
    correct_indices = torch.arange(batch_size, device=similarity_matrix.device)
    
    # Sort similarities in descending order
    sorted_indices = torch.argsort(similarity_matrix, dim=1, descending=True)
    
    metrics = {}
    
    # Compute recall@k for each k value
    for k in k_values:
        # Check if correct match is in top-k
        top_k_indices = sorted_indices[:, :k]
        hits = (top_k_indices == correct_indices.unsqueeze(1)).any(dim=1)
        recall_at_k = hits.float().mean().item()
        metrics[f"recall_at_{k}"] = recall_at_k
    
    # Compute median rank
    ranks = []
    for i in range(batch_size):
        rank = (sorted_indices[i] == correct_indices[i]).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)
    
    metrics["median_rank"] = np.median(ranks)
    
    # Compute mean average precision
    ap_scores = []
    for i in range(batch_size):
        sorted_similarities = similarity_matrix[i][sorted_indices[i]]
        correct_mask = (sorted_indices[i] == correct_indices[i]).float()
        
        # Compute precision at each position
        precision_at_k = []
        for k in range(1, batch_size + 1):
            precision = correct_mask[:k].sum() / k
            precision_at_k.append(precision)
        
        # Compute average precision
        ap = sum(precision_at_k) / batch_size
        ap_scores.append(ap.item())
    
    metrics["mean_average_precision"] = np.mean(ap_scores)
    
    return metrics


def format_time(seconds: float) -> str:
    """Format time duration in a human-readable format.
    
    Args:
        seconds: Time duration in seconds.
        
    Returns:
        Formatted time string.
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.0,
        mode: str = "min",
        restore_best_weights: bool = True
    ):
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping.
            min_delta: Minimum change to qualify as an improvement.
            mode: "min" for loss, "max" for accuracy.
            restore_best_weights: Whether to restore best weights when stopping.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, score: float, model: nn.Module) -> bool:
        """Check if training should stop.
        
        Args:
            score: Current score to evaluate.
            model: Model to potentially save weights from.
            
        Returns:
            True if training should stop, False otherwise.
        """
        if self.best_score is None:
            self.best_score = score
            self.best_weights = model.state_dict().copy()
            return False
        
        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
            self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        
        return False
