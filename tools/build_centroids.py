
import sys
import numpy as np
import pickle
import argparse
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path

# Add project root to path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from food2recipe.core.settings import load_settings
from food2recipe.core.logging_utils import setup_logger
from food2recipe.retrieval.index_faiss import RetrievalIndex

logger = setup_logger("build_centroids")

def main():
    settings = load_settings()
    
    # 1. Load Index
    logger.info("Loading index...")
    idx_wrapper = RetrievalIndex(settings)
    index_path = settings.ARTIFACTS_DIR / "index"
    try:
        idx_wrapper.load(index_path)
    except FileNotFoundError:
        logger.error("Index not found. Please run 'python -m food2recipe.scripts.build_index' first.")
        sys.exit(1)

    metadata = idx_wrapper.metadata
    
    # 2. Extract Vectors and Group by Class
    # We need to handle both FAISS and Numpy backends
    
    logger.info("Computing centroids...")
    
    # Detect dimension
    if idx_wrapper.is_faiss:
        # FAISS
        try:
            # Check if reconstruct is supported
            # IndexFlatIP supports it
            dim = idx_wrapper.index.d
        except Exception:
            logger.error("Could not determine dimension from FAISS index.")
            sys.exit(1)
    else:
        # Numpy
        dim = idx_wrapper.index.shape[1]

    class_vectors = defaultdict(list)
    
    ntotal = len(metadata)
    
    for i in tqdm(range(ntotal)):
        meta = metadata[i]
        food_name = meta['food_name']
        
        # Get vector
        if idx_wrapper.is_faiss:
            # FAISS reconstruct
            # Note: reconstruct returns float32 array
            vec = idx_wrapper.index.reconstruct(i)
        else:
            # Numpy
            vec = idx_wrapper.index[i]
            
        class_vectors[food_name].append(vec)

    # 3. Compute Mean
    centroids = {}
    for food_name, vecs in class_vectors.items():
        # Stack and mean
        mat = np.vstack(vecs)
        # Normalize? Yes, usually better for cosine sim
        mean_vec = np.mean(mat, axis=0)
        # L2 normalize the centroid
        norm = np.linalg.norm(mean_vec)
        if norm > 1e-6:
            mean_vec = mean_vec / norm
            
        centroids[food_name] = mean_vec
        
    logger.info(f"Computed centroids for {len(centroids)} classes.")
    
    # 4. Save
    save_path = settings.ARTIFACTS_DIR / "class_centroids.npy"
    np.save(save_path, centroids)
    logger.info(f"Saved centroids to {save_path}")

if __name__ == "__main__":
    main()
