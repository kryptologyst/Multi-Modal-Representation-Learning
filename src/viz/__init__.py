"""Visualization utilities for multi-modal representation learning."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

# Set style
plt.style.use('default')
sns.set_palette("husl")


class MultiModalVisualizer:
    """Visualizer for multi-modal representation learning results."""
    
    def __init__(self, config: DictConfig):
        """Initialize visualizer.
        
        Args:
            config: Configuration object.
        """
        self.config = config
        self.save_dir = config.get("asset_dir", "assets")
        
        logger.info(f"Initialized MultiModalVisualizer with save_dir: {self.save_dir}")
    
    def visualize_embeddings(
        self,
        embeddings: torch.Tensor,
        labels: Optional[List[str]] = None,
        title: str = "Embedding Visualization",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Visualize embeddings using t-SNE or PCA.
        
        Args:
            embeddings: Embeddings tensor.
            labels: Optional labels for coloring.
            title: Plot title.
            save_path: Optional path to save the plot.
            
        Returns:
            Matplotlib figure.
        """
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        
        # Convert to numpy
        embeddings_np = embeddings.detach().cpu().numpy()
        
        # Reduce dimensionality
        if embeddings_np.shape[1] > 50:
            # Use PCA first for high-dimensional data
            pca = PCA(n_components=50)
            embeddings_np = pca.fit_transform(embeddings_np)
        
        # Use t-SNE for final visualization
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings_np)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if labels is not None:
            unique_labels = list(set(labels))
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
            
            for i, label in enumerate(unique_labels):
                mask = [l == label for l in labels]
                ax.scatter(
                    embeddings_2d[mask, 0],
                    embeddings_2d[mask, 1],
                    c=[colors[i]],
                    label=label,
                    alpha=0.7,
                    s=50
                )
            ax.legend()
        else:
            ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7, s=50)
        
        ax.set_title(title)
        ax.set_xlabel("t-SNE Component 1")
        ax.set_ylabel("t-SNE Component 2")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Embedding visualization saved to {save_path}")
        
        return fig
    
    def visualize_similarity_matrix(
        self,
        similarity_matrix: torch.Tensor,
        image_labels: Optional[List[str]] = None,
        text_labels: Optional[List[str]] = None,
        title: str = "Similarity Matrix",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Visualize similarity matrix as heatmap.
        
        Args:
            similarity_matrix: Similarity matrix tensor.
            image_labels: Optional labels for image axis.
            text_labels: Optional labels for text axis.
            title: Plot title.
            save_path: Optional path to save the plot.
            
        Returns:
            Matplotlib figure.
        """
        # Convert to numpy
        similarity_np = similarity_matrix.detach().cpu().numpy()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        
        im = ax.imshow(similarity_np, cmap="viridis", aspect="auto")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Similarity Score")
        
        # Set labels
        if image_labels:
            ax.set_yticks(range(len(image_labels)))
            ax.set_yticklabels(image_labels, rotation=0)
        if text_labels:
            ax.set_xticks(range(len(text_labels)))
            ax.set_xticklabels(text_labels, rotation=45, ha="right")
        
        ax.set_title(title)
        ax.set_xlabel("Text")
        ax.set_ylabel("Image")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Similarity matrix visualization saved to {save_path}")
        
        return fig
    
    def visualize_retrieval_results(
        self,
        query_image: Image.Image,
        query_text: str,
        retrieved_images: List[Image.Image],
        retrieved_texts: List[str],
        similarity_scores: List[float],
        title: str = "Retrieval Results",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Visualize retrieval results.
        
        Args:
            query_image: Query image.
            query_text: Query text.
            retrieved_images: Retrieved images.
            retrieved_texts: Retrieved texts.
            similarity_scores: Similarity scores.
            title: Plot title.
            save_path: Optional path to save the plot.
            
        Returns:
            Matplotlib figure.
        """
        num_results = len(retrieved_images)
        fig, axes = plt.subplots(2, num_results + 1, figsize=(4 * (num_results + 1), 8))
        
        if num_results == 0:
            axes = axes.reshape(2, 1)
        
        # Show query
        axes[0, 0].imshow(query_image)
        axes[0, 0].set_title(f"Query Image\n{query_text}", fontsize=10)
        axes[0, 0].axis("off")
        
        axes[1, 0].text(0.5, 0.5, f"Query: {query_text}", 
                       ha="center", va="center", fontsize=12, weight="bold")
        axes[1, 0].axis("off")
        
        # Show retrieved results
        for i in range(num_results):
            axes[0, i + 1].imshow(retrieved_images[i])
            axes[0, i + 1].set_title(f"Result {i + 1}\nScore: {similarity_scores[i]:.3f}", 
                                    fontsize=10)
            axes[0, i + 1].axis("off")
            
            axes[1, i + 1].text(0.5, 0.5, retrieved_texts[i], 
                               ha="center", va="center", fontsize=10, wrap=True)
            axes[1, i + 1].axis("off")
        
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Retrieval results visualization saved to {save_path}")
        
        return fig
    
    def visualize_attention_maps(
        self,
        attention_weights: torch.Tensor,
        image: Image.Image,
        text_tokens: List[str],
        title: str = "Attention Maps",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Visualize attention maps for image-text pairs.
        
        Args:
            attention_weights: Attention weights tensor.
            image: PIL Image object.
            text_tokens: List of text tokens.
            title: Plot title.
            save_path: Optional path to save the plot.
            
        Returns:
            Matplotlib figure.
        """
        # Convert attention weights to numpy
        attention_np = attention_weights.detach().cpu().numpy()
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Show image
        ax1.imshow(image)
        ax1.set_title("Input Image")
        ax1.axis("off")
        
        # Show attention heatmap
        sns.heatmap(
            attention_np,
            xticklabels=text_tokens,
            yticklabels=False,
            cmap="Blues",
            ax=ax2,
            cbar_kws={"label": "Attention Weight"}
        )
        ax2.set_title("Attention Weights")
        ax2.set_xlabel("Text Tokens")
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Attention maps visualization saved to {save_path}")
        
        return fig
    
    def create_retrieval_dashboard(
        self,
        metrics: Dict[str, float],
        examples: List[Dict[str, Any]],
        save_path: Optional[str] = None
    ) -> go.Figure:
        """Create interactive retrieval dashboard.
        
        Args:
            metrics: Dictionary of evaluation metrics.
            examples: List of retrieval examples.
            save_path: Optional path to save the dashboard.
            
        Returns:
            Plotly figure.
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Retrieval Metrics", "Similarity Distribution", 
                          "Top Matches", "Performance by Category"),
            specs=[[{"type": "bar"}, {"type": "histogram"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # Plot metrics
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        fig.add_trace(
            go.Bar(x=metric_names, y=metric_values, name="Metrics"),
            row=1, col=1
        )
        
        # Plot similarity distribution
        similarities = [ex["similarity_score"] for ex in examples]
        fig.add_trace(
            go.Histogram(x=similarities, name="Similarity Distribution"),
            row=1, col=2
        )
        
        # Plot top matches
        ranks = [match["rank"] for ex in examples for match in ex["top_matches"]]
        similarities_top = [match["similarity"] for ex in examples for match in ex["top_matches"]]
        
        fig.add_trace(
            go.Scatter(x=ranks, y=similarities_top, mode="markers", name="Top Matches"),
            row=2, col=1
        )
        
        # Plot performance by category
        categories = {}
        for ex in examples:
            cat = ex["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(ex["similarity_score"])
        
        cat_names = list(categories.keys())
        cat_scores = [np.mean(categories[cat]) for cat in cat_names]
        
        fig.add_trace(
            go.Bar(x=cat_names, y=cat_scores, name="Category Performance"),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="Multi-Modal Retrieval Dashboard",
            showlegend=False,
            height=800
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Retrieval dashboard saved to {save_path}")
        
        return fig
    
    def save_qualitative_results(
        self,
        results: List[Dict[str, Any]],
        save_dir: str
    ) -> None:
        """Save qualitative results to files.
        
        Args:
            results: List of result dictionaries.
            save_dir: Directory to save results.
        """
        import json
        from pathlib import Path
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save results as JSON
        results_file = save_path / "qualitative_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Qualitative results saved to {results_file}")


def create_visualizer(config: DictConfig) -> MultiModalVisualizer:
    """Create visualizer based on configuration.
    
    Args:
        config: Configuration object.
        
    Returns:
        Initialized visualizer.
    """
    return MultiModalVisualizer(config)


def visualize_embeddings_2d(
    embeddings: torch.Tensor,
    labels: Optional[List[str]] = None,
    method: str = "tsne",
    save_path: Optional[str] = None
) -> plt.Figure:
    """Create 2D visualization of embeddings.
    
    Args:
        embeddings: Embeddings tensor.
        labels: Optional labels for coloring.
        method: Dimensionality reduction method ("tsne" or "pca").
        save_path: Optional path to save the plot.
        
    Returns:
        Matplotlib figure.
    """
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    
    # Convert to numpy
    embeddings_np = embeddings.detach().cpu().numpy()
    
    # Reduce dimensionality
    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42)
    else:
        reducer = PCA(n_components=2)
    
    embeddings_2d = reducer.fit_transform(embeddings_np)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if labels is not None:
        unique_labels = list(set(labels))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = [l == label for l in labels]
            ax.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=[colors[i]],
                label=label,
                alpha=0.7,
                s=50
            )
        ax.legend()
    else:
        ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7, s=50)
    
    ax.set_title(f"Embedding Visualization ({method.upper()})")
    ax.set_xlabel(f"{method.upper()} Component 1")
    ax.set_ylabel(f"{method.upper()} Component 2")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig
