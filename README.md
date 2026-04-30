# Multi-Modal Representation Learning

A research-ready implementation of multi-modal representation learning using CLIP-style models for cross-modal retrieval and alignment.

## Overview

This project implements a comprehensive framework for learning joint representations of images and text through contrastive learning. The system enables cross-modal retrieval, where you can find similar images given text queries and vice versa.

### Key Features

- **CLIP-style Architecture**: Dual encoder design with vision and text encoders
- **Contrastive Learning**: InfoNCE loss for learning aligned representations
- **Cross-Modal Retrieval**: Text-to-image and image-to-text search capabilities
- **Comprehensive Evaluation**: Multiple retrieval metrics (Recall@K, mAP, median rank)
- **Interactive Demo**: Streamlit-based web interface for exploration
- **Modern Stack**: PyTorch 2.x, Transformers, Hydra configs, type hints
- **Reproducible**: Deterministic seeding, checkpointing, logging

## Project Structure

```
├── src/                    # Source code
│   ├── data/              # Data loading and preprocessing
│   ├── models/            # Model architectures
│   ├── losses/            # Loss functions
│   ├── eval/              # Evaluation utilities
│   ├── viz/               # Visualization tools
│   └── utils/             # Utility functions
├── configs/                # Configuration files
│   ├── model/             # Model configurations
│   ├── train/             # Training configurations
│   ├── eval/              # Evaluation configurations
│   └── data/              # Data configurations
├── scripts/               # Training and evaluation scripts
├── demo/                  # Demo applications
├── tests/                 # Unit tests
├── data/                  # Dataset directory
├── assets/                # Generated assets
├── checkpoints/           # Model checkpoints
├── logs/                  # Training logs
└── outputs/               # Output files
```

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended) or Apple Silicon (MPS) or CPU

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kryptologyst/Multi-Modal-Representation-Learning.git
   cd Multi-Modal-Representation-Learning
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   Or for development:
   ```bash
   pip install -e ".[dev]"
   ```

4. **Install pre-commit hooks (optional):**
   ```bash
   pre-commit install
   ```

## Quick Start

### 1. Training

Train the model on the toy dataset:

```bash
python scripts/train.py --config configs/config.yaml
```

### 2. Evaluation

Evaluate the trained model:

```bash
python scripts/train.py --config configs/config.yaml --eval-only
```

### 3. Interactive Demo

Launch the Streamlit demo:

```bash
streamlit run demo/streamlit_demo.py
```

## Usage

### Configuration

The project uses Hydra for configuration management. Key configuration files:

- `configs/config.yaml`: Main configuration
- `configs/model/clip_base.yaml`: Model architecture
- `configs/train/default.yaml`: Training parameters
- `configs/eval/default.yaml`: Evaluation settings
- `configs/data/toy_dataset.yaml`: Dataset configuration

### Training

```python
from omegaconf import OmegaConf
from src.models import CLIPModel
from src.data import create_data_loaders
from src.losses import MultiModalLoss

# Load configuration
config = OmegaConf.load("configs/config.yaml")

# Create data loaders
train_loader, val_loader, test_loader = create_data_loaders(config.data)

# Initialize model
model = CLIPModel(config.model)

# Initialize loss
criterion = MultiModalLoss(config.model)

# Training loop
for epoch in range(config.train.num_epochs):
    for batch in train_loader:
        # Forward pass
        outputs = model(batch["pixel_values"], batch["input_ids"], batch["attention_mask"])
        losses = criterion(outputs)
        
        # Backward pass
        losses["total_loss"].backward()
        optimizer.step()
```

### Evaluation

```python
from src.eval import MultiModalEvaluator

# Initialize evaluator
evaluator = MultiModalEvaluator(config.eval)

# Evaluate model
metrics = evaluator.evaluate(model, test_loader, device)

print(f"Recall@1: {metrics['recall_at_1']:.3f}")
print(f"Recall@5: {metrics['recall_at_5']:.3f}")
print(f"mAP: {metrics['mean_average_precision']:.3f}")
```

### Cross-Modal Retrieval

```python
from transformers import CLIPProcessor

# Initialize processor
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Text-to-image retrieval
text_query = "A cute dog playing in the park"
text_inputs = processor(text=[text_query], return_tensors="pt")
text_embeddings = model.encode_text(text_inputs["input_ids"], text_inputs["attention_mask"])

# Compute similarities with image embeddings
similarities = torch.matmul(text_embeddings, image_embeddings.T)
top_indices = torch.topk(similarities, k=5).indices
```

## Dataset

### Toy Dataset

The project includes a synthetic toy dataset for demonstration purposes:

- **Categories**: Animals, vehicles, nature, food, objects
- **Samples**: 1000 image-text pairs
- **Splits**: 70% train, 15% validation, 15% test
- **Format**: Synthetic images with text descriptions

### Custom Dataset

To use your own dataset:

1. **Prepare data structure:**
   ```
   data/
   ├── images/
   │   ├── image1.jpg
   │   ├── image2.jpg
   │   └── ...
   └── annotations.json
   ```

2. **Create annotations file:**
   ```json
   [
     {
       "image_path": "image1.jpg",
       "text": "A description of the image",
       "category": "category_name",
       "split": "train"
     }
   ]
   ```

3. **Update configuration:**
   ```yaml
   data:
     _target_: src.data.ImageTextPairDataset
     data_path: "data"
     # ... other parameters
   ```

## Model Architecture

### CLIP-Style Dual Encoder

- **Vision Encoder**: ViT-Base (patch size 32, 12 layers)
- **Text Encoder**: Transformer (12 layers, 8 heads)
- **Projection**: Linear layers to shared embedding space
- **Temperature**: Learnable temperature parameter for contrastive learning

### Loss Functions

- **Contrastive Loss**: InfoNCE objective for alignment
- **Hard Negative Mining**: Improved contrastive learning
- **Multi-Modal Loss**: Combined loss components

## Evaluation Metrics

### Retrieval Metrics

- **Recall@K**: Fraction of queries with correct match in top-K
- **Mean Average Precision (mAP)**: Average precision across all queries
- **Median Rank**: Median rank of correct matches
- **CLIP Score**: Average cosine similarity

### Cross-Modal Evaluation

- **Image-to-Text**: Retrieve text descriptions for images
- **Text-to-Image**: Retrieve images for text queries
- **Bidirectional**: Both directions evaluated

## Visualization

### Embedding Visualization

- **t-SNE**: 2D visualization of learned embeddings
- **PCA**: Principal component analysis
- **Category Coloring**: Color by semantic categories

### Retrieval Results

- **Similarity Matrix**: Heatmap of cross-modal similarities
- **Attention Maps**: Visualization of attention weights
- **Retrieval Examples**: Qualitative results with scores

## Demo Application

The Streamlit demo provides an interactive interface for:

- **Text-to-Image Search**: Enter text queries to find similar images
- **Image-to-Text Search**: Upload images to find similar descriptions
- **Dataset Exploration**: Browse the dataset and view statistics
- **Model Information**: View model parameters and performance

### Running the Demo

```bash
streamlit run demo/streamlit_demo.py
```

## Development

### Code Quality

- **Type Hints**: Full type annotation coverage
- **Documentation**: Google/NumPy docstring format
- **Formatting**: Black code formatting
- **Linting**: Ruff for code quality
- **Testing**: pytest for unit tests

### Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files
```

### Testing

```bash
pytest tests/
```

## Configuration

### Model Configuration

```yaml
model:
  model_name: "openai/clip-vit-base-patch32"
  embedding_dim: 512
  temperature: 0.07
  learning_rate: 1e-4
  weight_decay: 0.01
```

### Training Configuration

```yaml
train:
  batch_size: 32
  num_epochs: 10
  learning_rate: 1e-4
  warmup_steps: 1000
  use_amp: true
  early_stopping_patience: 5
```

### Evaluation Configuration

```yaml
eval:
  metrics:
    - "recall_at_1"
    - "recall_at_5"
    - "recall_at_10"
    - "mean_average_precision"
    - "median_rank"
  top_k_values: [1, 5, 10, 50, 100]
```

## Safety and Limitations

### Safety Disclaimer

This is a research/educational project for multi-modal representation learning. The model may:

- Generate biased or inappropriate content
- Have limitations in understanding complex scenes
- Not be suitable for production use without proper evaluation

### Limitations

- **Dataset Bias**: Performance depends on training data quality
- **Domain Gap**: May not generalize to unseen domains
- **Computational Requirements**: Requires GPU for efficient training
- **Evaluation**: Metrics may not capture all aspects of quality

### Responsible Use

- Use appropriate content filters
- Evaluate on diverse datasets
- Consider bias and fairness implications
- Follow ethical AI guidelines

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{multimodal_representation_learning,
  title={Multi-Modal Representation Learning},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Multi-Modal-Representation-Learning}
}
```

## Acknowledgments

- OpenAI CLIP team for the original CLIP model
- Hugging Face for the Transformers library
- The open-source community for various tools and libraries

## Contact

For questions and support, please open an issue on GitHub or contact [github.com/kryptologyst](https://github.com/kryptologyst).
# Multi-Modal-Representation-Learning
