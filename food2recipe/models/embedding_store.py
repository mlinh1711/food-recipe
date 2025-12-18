# File: food2recipe/models/embedding_store.py
import torch
import pickle
from pathlib import Path
from food2recipe.core.logging_utils import setup_logger

logger = setup_logger("embedding_store")

def save_embeddings(embeddings, metadata, output_path: Path):
    """
    Saves embeddings (Tensor) and metadata (List of dict/df) to disk.
    """
    data = {
        "embeddings": embeddings,
        "metadata": metadata
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(data, f)
    logger.info(f"Saved {len(metadata)} embeddings to {output_path}")

def load_embeddings(input_path: Path):
    if not input_path.exists():
        raise FileNotFoundError(f"No embedding store at {input_path}")
        
    with open(input_path, "rb") as f:
        data = pickle.load(f)
    return data["embeddings"], data["metadata"]
