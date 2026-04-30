"""Streamlit demo for multi-modal representation learning."""

import streamlit as st
import torch
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.models import CLIPModel
from src.data import ToyDataset
from src.utils import get_device, set_seed
from src.viz import MultiModalVisualizer
from omegaconf import OmegaConf

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Multi-Modal Representation Learning Demo",
    page_icon="🔍",
    layout="wide"
)

# Safety disclaimer
st.sidebar.markdown("""
## ⚠️ Safety Disclaimer

This is a research/educational demo for multi-modal representation learning. 
The model may generate biased or inappropriate content. Use responsibly.

**Not for production use.**
""")

# Title
st.title("🔍 Multi-Modal Representation Learning Demo")
st.markdown("""
This demo showcases cross-modal retrieval using CLIP-style models. 
Upload an image or enter text to find similar content across modalities.
""")

# Initialize session state
if "model" not in st.session_state:
    st.session_state.model = None
if "dataset" not in st.session_state:
    st.session_state.dataset = None
if "device" not in st.session_state:
    st.session_state.device = None

@st.cache_resource
def load_model():
    """Load the CLIP model."""
    try:
        # Load config
        config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
        config = OmegaConf.load(config_path)
        
        # Set device
        device = get_device("auto")
        
        # Initialize model
        model = CLIPModel(config.model).to(device)
        model.eval()
        
        logger.info("Model loaded successfully")
        return model, device
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, None

@st.cache_resource
def load_dataset():
    """Load the toy dataset."""
    try:
        # Load config
        config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
        config = OmegaConf.load(config_path)
        
        # Create dataset
        dataset = ToyDataset(config.data, split="test")
        
        logger.info("Dataset loaded successfully")
        return dataset
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return None

def main():
    """Main demo function."""
    # Load model and dataset
    if st.session_state.model is None:
        with st.spinner("Loading model..."):
            st.session_state.model, st.session_state.device = load_model()
    
    if st.session_state.dataset is None:
        with st.spinner("Loading dataset..."):
            st.session_state.dataset = load_dataset()
    
    if st.session_state.model is None or st.session_state.dataset is None:
        st.error("Failed to load model or dataset. Please check the logs.")
        return
    
    model = st.session_state.model
    device = st.session_state.device
    dataset = st.session_state.dataset
    
    # Sidebar controls
    st.sidebar.header("🎛️ Controls")
    
    # Query type selection
    query_type = st.sidebar.selectbox(
        "Query Type",
        ["Text to Images", "Image to Texts", "Both"]
    )
    
    # Number of results
    num_results = st.sidebar.slider("Number of Results", 1, 10, 5)
    
    # Similarity threshold
    similarity_threshold = st.sidebar.slider("Similarity Threshold", 0.0, 1.0, 0.5)
    
    # Main content
    if query_type in ["Text to Images", "Both"]:
        st.header("📝 Text to Image Retrieval")
        
        # Text input
        text_query = st.text_area(
            "Enter your text query:",
            value="A cute dog playing in the park",
            height=100
        )
        
        if st.button("Search Images", key="text_search"):
            if text_query.strip():
                with st.spinner("Searching for similar images..."):
                    # Get text embedding
                    from transformers import CLIPProcessor
                    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                    
                    text_inputs = processor(text=[text_query], return_tensors="pt", padding=True)
                    text_embeddings = model.encode_text(
                        text_inputs["input_ids"].to(device),
                        text_inputs["attention_mask"].to(device)
                    )
                    
                    # Get image embeddings from dataset
                    image_embeddings = []
                    image_data = []
                    
                    for i in range(min(len(dataset), 100)):  # Limit for demo
                        sample = dataset[i]
                        image = sample["image"]
                        
                        # Process image
                        image_inputs = processor(images=[image], return_tensors="pt")
                        image_embedding = model.encode_image(image_inputs["pixel_values"].to(device))
                        
                        image_embeddings.append(image_embedding)
                        image_data.append({
                            "image": image,
                            "text": sample["text"],
                            "category": sample["category"],
                            "id": sample["id"]
                        })
                    
                    # Compute similarities
                    image_embeddings = torch.cat(image_embeddings, dim=0)
                    similarities = torch.matmul(text_embeddings, image_embeddings.T)
                    
                    # Get top results
                    top_scores, top_indices = torch.topk(similarities, k=num_results, dim=-1)
                    
                    # Display results
                    st.subheader("🎯 Search Results")
                    
                    cols = st.columns(min(num_results, 3))
                    for i, (score, idx) in enumerate(zip(top_scores[0], top_indices[0])):
                        if score.item() >= similarity_threshold:
                            with cols[i % 3]:
                                st.image(image_data[idx]["image"], caption=f"Score: {score.item():.3f}")
                                st.write(f"**Text:** {image_data[idx]['text']}")
                                st.write(f"**Category:** {image_data[idx]['category']}")
                    
                    # Show similarity distribution
                    st.subheader("📊 Similarity Distribution")
                    fig = px.histogram(
                        x=similarities[0].cpu().numpy(),
                        nbins=20,
                        title="Similarity Scores Distribution",
                        labels={"x": "Similarity Score", "y": "Count"}
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    if query_type in ["Image to Texts", "Both"]:
        st.header("🖼️ Image to Text Retrieval")
        
        # Image upload
        uploaded_file = st.file_uploader(
            "Upload an image:",
            type=["jpg", "jpeg", "png", "bmp", "tiff"],
            help="Upload an image to find similar text descriptions"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", width=300)
            
            if st.button("Search Texts", key="image_search"):
                with st.spinner("Searching for similar texts..."):
                    # Get image embedding
                    from transformers import CLIPProcessor
                    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                    
                    image_inputs = processor(images=[image], return_tensors="pt")
                    image_embeddings = model.encode_image(image_inputs["pixel_values"].to(device))
                    
                    # Get text embeddings from dataset
                    text_embeddings = []
                    text_data = []
                    
                    for i in range(min(len(dataset), 100)):  # Limit for demo
                        sample = dataset[i]
                        text = sample["text"]
                        
                        # Process text
                        text_inputs = processor(text=[text], return_tensors="pt", padding=True)
                        text_embedding = model.encode_text(
                            text_inputs["input_ids"].to(device),
                            text_inputs["attention_mask"].to(device)
                        )
                        
                        text_embeddings.append(text_embedding)
                        text_data.append({
                            "text": text,
                            "category": sample["category"],
                            "id": sample["id"]
                        })
                    
                    # Compute similarities
                    text_embeddings = torch.cat(text_embeddings, dim=0)
                    similarities = torch.matmul(image_embeddings, text_embeddings.T)
                    
                    # Get top results
                    top_scores, top_indices = torch.topk(similarities, k=num_results, dim=-1)
                    
                    # Display results
                    st.subheader("🎯 Search Results")
                    
                    for i, (score, idx) in enumerate(zip(top_scores[0], top_indices[0])):
                        if score.item() >= similarity_threshold:
                            st.write(f"**Rank {i+1}:** {text_data[idx]['text']}")
                            st.write(f"**Score:** {score.item():.3f} | **Category:** {text_data[idx]['category']}")
                            st.write("---")
    
    # Dataset exploration
    st.header("📚 Dataset Exploration")
    
    # Show dataset statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Samples", len(dataset))
    
    with col2:
        categories = list(set(sample["category"] for sample in dataset.data))
        st.metric("Categories", len(categories))
    
    with col3:
        avg_text_length = np.mean([len(sample["text"]) for sample in dataset.data])
        st.metric("Avg Text Length", f"{avg_text_length:.1f}")
    
    # Show category distribution
    st.subheader("📊 Category Distribution")
    category_counts = {}
    for sample in dataset.data:
        category = sample["category"]
        category_counts[category] = category_counts.get(category, 0) + 1
    
    fig = px.pie(
        values=list(category_counts.values()),
        names=list(category_counts.keys()),
        title="Dataset Category Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Show sample data
    st.subheader("🔍 Sample Data")
    
    if st.button("Show Random Samples"):
        # Get random samples
        indices = np.random.choice(len(dataset), size=min(6, len(dataset)), replace=False)
        
        cols = st.columns(3)
        for i, idx in enumerate(indices):
            sample = dataset[idx]
            with cols[i % 3]:
                st.image(sample["image"], caption=sample["text"], width=200)
                st.write(f"**Category:** {sample['category']}")
    
    # Model information
    st.sidebar.header("🤖 Model Info")
    st.sidebar.write(f"**Model:** CLIP ViT-Base")
    st.sidebar.write(f"**Device:** {device}")
    st.sidebar.write(f"**Parameters:** {model.count_parameters()['total']:,}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        Multi-Modal Representation Learning Demo | 
        <a href='https://github.com/kryptologyst' target='_blank'>github.com/kryptologyst</a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
