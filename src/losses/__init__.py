"""Loss functions for multi-modal representation learning."""

import logging
from typing import Dict, Optional, Tuple
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class ContrastiveLoss(nn.Module):
    """Contrastive loss for multi-modal representation learning.
    
    This loss encourages similar image-text pairs to have high similarity
    and dissimilar pairs to have low similarity.
    """
    
    def __init__(self, temperature: float = 0.07, weight: float = 1.0):
        """Initialize contrastive loss.
        
        Args:
            temperature: Temperature parameter for scaling logits.
            weight: Weight for this loss component.
        """
        super().__init__()
        self.temperature = temperature
        self.weight = weight
        
        logger.info(f"Initialized ContrastiveLoss with temperature={temperature}, weight={weight}")
    
    def forward(
        self,
        logits_per_image: torch.Tensor,
        logits_per_text: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute contrastive loss.
        
        Args:
            logits_per_image: Logits for image-to-text similarity.
            logits_per_text: Logits for text-to-image similarity.
            labels: Optional labels for supervised learning.
            
        Returns:
            Dictionary containing loss components.
        """
        batch_size = logits_per_image.size(0)
        
        if labels is None:
            # Self-supervised: diagonal elements are positive pairs
            labels = torch.arange(batch_size, device=logits_per_image.device)
        
        # Compute cross-entropy loss
        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)
        
        # Total contrastive loss
        total_loss = (loss_i2t + loss_t2i) / 2
        
        return {
            "contrastive_loss": total_loss * self.weight,
            "loss_i2t": loss_i2t,
            "loss_t2i": loss_t2i,
            "total_loss": total_loss * self.weight
        }


class TripletLoss(nn.Module):
    """Triplet loss for multi-modal representation learning.
    
    This loss encourages the distance between positive pairs to be smaller
    than the distance between negative pairs by a margin.
    """
    
    def __init__(self, margin: float = 1.0, weight: float = 1.0):
        """Initialize triplet loss.
        
        Args:
            margin: Margin for triplet loss.
            weight: Weight for this loss component.
        """
        super().__init__()
        self.margin = margin
        self.weight = weight
        
        logger.info(f"Initialized TripletLoss with margin={margin}, weight={weight}")
    
    def forward(
        self,
        anchor_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute triplet loss.
        
        Args:
            anchor_embeddings: Anchor embeddings.
            positive_embeddings: Positive embeddings.
            negative_embeddings: Negative embeddings.
            
        Returns:
            Dictionary containing loss components.
        """
        # Compute distances
        pos_dist = F.pairwise_distance(anchor_embeddings, positive_embeddings)
        neg_dist = F.pairwise_distance(anchor_embeddings, negative_embeddings)
        
        # Compute triplet loss
        loss = F.relu(pos_dist - neg_dist + self.margin)
        loss = loss.mean()
        
        return {
            "triplet_loss": loss * self.weight,
            "total_loss": loss * self.weight
        }


class InfoNCELoss(nn.Module):
    """InfoNCE loss for multi-modal representation learning.
    
    This is a variant of contrastive loss that uses the InfoNCE objective.
    """
    
    def __init__(self, temperature: float = 0.07, weight: float = 1.0):
        """Initialize InfoNCE loss.
        
        Args:
            temperature: Temperature parameter for scaling logits.
            weight: Weight for this loss component.
        """
        super().__init__()
        self.temperature = temperature
        self.weight = weight
        
        logger.info(f"Initialized InfoNCELoss with temperature={temperature}, weight={weight}")
    
    def forward(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute InfoNCE loss.
        
        Args:
            image_embeddings: Image embeddings.
            text_embeddings: Text embeddings.
            
        Returns:
            Dictionary containing loss components.
        """
        batch_size = image_embeddings.size(0)
        
        # Normalize embeddings
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(image_embeddings, text_embeddings.T) / self.temperature
        
        # Create labels (diagonal elements are positive pairs)
        labels = torch.arange(batch_size, device=image_embeddings.device)
        
        # Compute InfoNCE loss
        loss_i2t = F.cross_entropy(similarity_matrix, labels)
        loss_t2i = F.cross_entropy(similarity_matrix.T, labels)
        
        total_loss = (loss_i2t + loss_t2i) / 2
        
        return {
            "infonce_loss": total_loss * self.weight,
            "loss_i2t": loss_i2t,
            "loss_t2i": loss_t2i,
            "total_loss": total_loss * self.weight
        }


class MultiModalLoss(nn.Module):
    """Combined loss function for multi-modal representation learning."""
    
    def __init__(self, config: DictConfig):
        """Initialize multi-modal loss.
        
        Args:
            config: Configuration object containing loss parameters.
        """
        super().__init__()
        self.config = config
        
        # Initialize loss components
        self.contrastive_loss = ContrastiveLoss(
            temperature=config.temperature,
            weight=config.contrastive_loss_weight
        )
        
        # Optional additional losses
        if hasattr(config, 'use_triplet_loss') and config.use_triplet_loss:
            self.triplet_loss = TripletLoss(
                margin=config.triplet_margin,
                weight=config.triplet_loss_weight
            )
        else:
            self.triplet_loss = None
        
        if hasattr(config, 'use_infonce_loss') and config.use_infonce_loss:
            self.infonce_loss = InfoNCELoss(
                temperature=config.temperature,
                weight=config.infonce_loss_weight
            )
        else:
            self.infonce_loss = None
        
        logger.info("Initialized MultiModalLoss with configured components")
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute combined loss.
        
        Args:
            outputs: Model outputs dictionary.
            labels: Optional labels for supervised learning.
            
        Returns:
            Dictionary containing all loss components.
        """
        losses = {}
        total_loss = 0.0
        
        # Contrastive loss
        if "logits_per_image" in outputs and "logits_per_text" in outputs:
            contrastive_losses = self.contrastive_loss(
                outputs["logits_per_image"],
                outputs["logits_per_text"],
                labels
            )
            losses.update(contrastive_losses)
            total_loss += contrastive_losses["contrastive_loss"]
        
        # InfoNCE loss
        if self.infonce_loss is not None and "image_embeddings" in outputs and "text_embeddings" in outputs:
            infonce_losses = self.infonce_loss(
                outputs["image_embeddings"],
                outputs["text_embeddings"]
            )
            losses.update(infonce_losses)
            total_loss += infonce_losses["infonce_loss"]
        
        # Triplet loss (requires additional negative samples)
        if self.triplet_loss is not None and "anchor_embeddings" in outputs:
            triplet_losses = self.triplet_loss(
                outputs["anchor_embeddings"],
                outputs["positive_embeddings"],
                outputs["negative_embeddings"]
            )
            losses.update(triplet_losses)
            total_loss += triplet_losses["triplet_loss"]
        
        losses["total_loss"] = total_loss
        
        return losses


class HardNegativeMiningLoss(nn.Module):
    """Hard negative mining loss for improved contrastive learning."""
    
    def __init__(self, temperature: float = 0.07, hard_negative_ratio: float = 0.5):
        """Initialize hard negative mining loss.
        
        Args:
            temperature: Temperature parameter for scaling logits.
            hard_negative_ratio: Ratio of hard negatives to use.
        """
        super().__init__()
        self.temperature = temperature
        self.hard_negative_ratio = hard_negative_ratio
        
        logger.info(f"Initialized HardNegativeMiningLoss with temperature={temperature}, ratio={hard_negative_ratio}")
    
    def forward(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute hard negative mining loss.
        
        Args:
            image_embeddings: Image embeddings.
            text_embeddings: Text embeddings.
            
        Returns:
            Dictionary containing loss components.
        """
        batch_size = image_embeddings.size(0)
        
        # Normalize embeddings
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(image_embeddings, text_embeddings.T) / self.temperature
        
        # Create labels
        labels = torch.arange(batch_size, device=image_embeddings.device)
        
        # Compute hard negatives for image-to-text
        i2t_similarities = similarity_matrix
        i2t_hard_negatives = self._get_hard_negatives(i2t_similarities, labels)
        
        # Compute hard negatives for text-to-image
        t2i_similarities = similarity_matrix.T
        t2i_hard_negatives = self._get_hard_negatives(t2i_similarities, labels)
        
        # Compute losses with hard negatives
        loss_i2t = F.cross_entropy(i2t_hard_negatives, labels)
        loss_t2i = F.cross_entropy(t2i_hard_negatives, labels)
        
        total_loss = (loss_i2t + loss_t2i) / 2
        
        return {
            "hard_negative_loss": total_loss,
            "loss_i2t": loss_i2t,
            "loss_t2i": loss_t2i,
            "total_loss": total_loss
        }
    
    def _get_hard_negatives(
        self,
        similarities: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Get hard negative samples.
        
        Args:
            similarities: Similarity matrix.
            labels: Ground truth labels.
            
        Returns:
            Hard negative similarities.
        """
        batch_size = similarities.size(0)
        
        # Get hard negative indices (highest similarity among negatives)
        hard_negative_indices = []
        
        for i in range(batch_size):
            # Get similarities for this sample
            sample_similarities = similarities[i]
            
            # Get negative indices (not the positive pair)
            negative_mask = torch.ones(batch_size, dtype=torch.bool, device=similarities.device)
            negative_mask[i] = False
            
            # Get hard negatives (top-k most similar negatives)
            num_hard_negatives = max(1, int(batch_size * self.hard_negative_ratio))
            hard_negatives = torch.topk(
                sample_similarities[negative_mask],
                k=num_hard_negatives,
                dim=0
            ).indices
            
            # Create hard negative similarities
            hard_negative_similarities = torch.zeros_like(sample_similarities)
            hard_negative_similarities[i] = sample_similarities[i]  # Keep positive
            hard_negative_similarities[negative_mask][hard_negatives] = sample_similarities[negative_mask][hard_negatives]
            
            hard_negative_indices.append(hard_negative_similarities)
        
        return torch.stack(hard_negative_indices)
