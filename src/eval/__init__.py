"""Evaluation utilities for multi-modal representation learning."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from omegaconf import DictConfig

from .utils import compute_similarity_matrix, compute_retrieval_metrics

logger = logging.getLogger(__name__)


class MultiModalEvaluator:
    """Evaluator for multi-modal representation learning models."""
    
    def __init__(self, config: DictConfig):
        """Initialize evaluator.
        
        Args:
            config: Configuration object containing evaluation parameters.
        """
        self.config = config
        self.metrics = config.metrics
        self.top_k_values = config.top_k_values
        
        logger.info(f"Initialized MultiModalEvaluator with metrics: {self.metrics}")
    
    def evaluate(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        device: torch.device
    ) -> Dict[str, float]:
        """Evaluate model on a dataset.
        
        Args:
            model: Model to evaluate.
            data_loader: Data loader for evaluation.
            device: Device to run evaluation on.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        model.eval()
        
        all_image_embeddings = []
        all_text_embeddings = []
        all_texts = []
        all_categories = []
        
        with torch.no_grad():
            for batch in data_loader:
                # Move batch to device
                pixel_values = batch["pixel_values"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                
                # Get embeddings
                embeddings = model.get_embeddings(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                all_image_embeddings.append(embeddings["image_embeddings"])
                all_text_embeddings.append(embeddings["text_embeddings"])
                all_texts.extend(batch["texts"])
                all_categories.extend(batch["categories"])
        
        # Concatenate all embeddings
        image_embeddings = torch.cat(all_image_embeddings, dim=0)
        text_embeddings = torch.cat(all_text_embeddings, dim=0)
        
        # Compute evaluation metrics
        metrics = {}
        
        if self.config.evaluate_cross_modal:
            # Image-to-text retrieval
            if self.config.image_to_text:
                i2t_metrics = self._evaluate_retrieval(
                    image_embeddings, text_embeddings, "image_to_text"
                )
                metrics.update(i2t_metrics)
            
            # Text-to-image retrieval
            if self.config.text_to_image:
                t2i_metrics = self._evaluate_retrieval(
                    text_embeddings, image_embeddings, "text_to_image"
                )
                metrics.update(t2i_metrics)
        
        # Compute additional metrics
        if "clip_score" in self.metrics:
            clip_score = self._compute_clip_score(image_embeddings, text_embeddings)
            metrics["clip_score"] = clip_score
        
        logger.info(f"Evaluation completed. Metrics: {list(metrics.keys())}")
        
        return metrics
    
    def _evaluate_retrieval(
        self,
        query_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        retrieval_type: str
    ) -> Dict[str, float]:
        """Evaluate retrieval performance.
        
        Args:
            query_embeddings: Query embeddings.
            candidate_embeddings: Candidate embeddings.
            retrieval_type: Type of retrieval ("image_to_text" or "text_to_image").
            
        Returns:
            Dictionary of retrieval metrics.
        """
        # Compute similarity matrix
        similarity_matrix = compute_similarity_matrix(
            query_embeddings, candidate_embeddings, normalize=True
        )
        
        # Compute retrieval metrics
        metrics = compute_retrieval_metrics(similarity_matrix, self.top_k_values)
        
        # Add prefix to metric names
        prefixed_metrics = {
            f"{retrieval_type}_{metric_name}": metric_value
            for metric_name, metric_value in metrics.items()
        }
        
        return prefixed_metrics
    
    def _compute_clip_score(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor
    ) -> float:
        """Compute CLIP score (cosine similarity).
        
        Args:
            image_embeddings: Image embeddings.
            text_embeddings: Text embeddings.
            
        Returns:
            Average CLIP score.
        """
        # Compute cosine similarity
        similarities = torch.sum(image_embeddings * text_embeddings, dim=-1)
        
        # Return average similarity
        return similarities.mean().item()
    
    def evaluate_retrieval_examples(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        device: torch.device,
        num_examples: int = 10
    ) -> List[Dict[str, Any]]:
        """Evaluate retrieval with specific examples.
        
        Args:
            model: Model to evaluate.
            data_loader: Data loader for evaluation.
            device: Device to run evaluation on.
            num_examples: Number of examples to evaluate.
            
        Returns:
            List of retrieval examples with results.
        """
        model.eval()
        
        examples = []
        count = 0
        
        with torch.no_grad():
            for batch in data_loader:
                if count >= num_examples:
                    break
                
                # Move batch to device
                pixel_values = batch["pixel_values"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                
                # Get embeddings
                embeddings = model.get_embeddings(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Compute similarities
                similarities = torch.matmul(
                    embeddings["image_embeddings"],
                    embeddings["text_embeddings"].T
                )
                
                # Get top matches
                for i in range(min(len(batch["texts"]), num_examples - count)):
                    example = {
                        "image_id": batch["ids"][i],
                        "text": batch["texts"][i],
                        "category": batch["categories"][i],
                        "similarity_score": similarities[i, i].item(),
                        "top_matches": []
                    }
                    
                    # Get top-5 matches
                    top_indices = torch.topk(similarities[i], k=5).indices
                    for j, idx in enumerate(top_indices):
                        example["top_matches"].append({
                            "rank": j + 1,
                            "text": batch["texts"][idx.item()],
                            "similarity": similarities[i, idx].item()
                        })
                    
                    examples.append(example)
                    count += 1
        
        return examples


class RetrievalEvaluator:
    """Specialized evaluator for retrieval tasks."""
    
    def __init__(self, top_k_values: List[int] = [1, 5, 10, 50, 100]):
        """Initialize retrieval evaluator.
        
        Args:
            top_k_values: List of k values for recall@k computation.
        """
        self.top_k_values = top_k_values
        
        logger.info(f"Initialized RetrievalEvaluator with top_k_values: {top_k_values}")
    
    def evaluate_retrieval(
        self,
        query_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        ground_truth: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Evaluate retrieval performance.
        
        Args:
            query_embeddings: Query embeddings.
            candidate_embeddings: Candidate embeddings.
            ground_truth: Optional ground truth labels.
            
        Returns:
            Dictionary of retrieval metrics.
        """
        # Compute similarity matrix
        similarity_matrix = compute_similarity_matrix(
            query_embeddings, candidate_embeddings, normalize=True
        )
        
        # Compute metrics
        metrics = compute_retrieval_metrics(similarity_matrix, self.top_k_values)
        
        return metrics
    
    def evaluate_cross_modal_retrieval(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor
    ) -> Dict[str, float]:
        """Evaluate cross-modal retrieval.
        
        Args:
            image_embeddings: Image embeddings.
            text_embeddings: Text embeddings.
            
        Returns:
            Dictionary of cross-modal retrieval metrics.
        """
        metrics = {}
        
        # Image-to-text retrieval
        i2t_metrics = self.evaluate_retrieval(image_embeddings, text_embeddings)
        for metric_name, metric_value in i2t_metrics.items():
            metrics[f"i2t_{metric_name}"] = metric_value
        
        # Text-to-image retrieval
        t2i_metrics = self.evaluate_retrieval(text_embeddings, image_embeddings)
        for metric_name, metric_value in t2i_metrics.items():
            metrics[f"t2i_{metric_name}"] = metric_value
        
        return metrics


class ClassificationEvaluator:
    """Evaluator for classification tasks."""
    
    def __init__(self, num_classes: int):
        """Initialize classification evaluator.
        
        Args:
            num_classes: Number of classes.
        """
        self.num_classes = num_classes
        
        logger.info(f"Initialized ClassificationEvaluator with {num_classes} classes")
    
    def evaluate_classification(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """Evaluate classification performance.
        
        Args:
            predictions: Model predictions.
            labels: Ground truth labels.
            
        Returns:
            Dictionary of classification metrics.
        """
        # Convert to numpy
        pred_np = predictions.cpu().numpy()
        label_np = labels.cpu().numpy()
        
        # Compute accuracy
        accuracy = accuracy_score(label_np, pred_np)
        
        # Compute precision, recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            label_np, pred_np, average="macro"
        )
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }


def create_evaluator(config: DictConfig) -> MultiModalEvaluator:
    """Create evaluator based on configuration.
    
    Args:
        config: Configuration object.
        
    Returns:
        Initialized evaluator.
    """
    return MultiModalEvaluator(config)


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    config: DictConfig,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate model and return metrics.
    
    Args:
        model: Model to evaluate.
        data_loader: Data loader for evaluation.
        config: Configuration object.
        device: Device to run evaluation on.
        
    Returns:
        Dictionary of evaluation metrics.
    """
    evaluator = create_evaluator(config)
    return evaluator.evaluate(model, data_loader, device)
