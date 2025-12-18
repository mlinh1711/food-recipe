# File: food2recipe/retrieval/index_faiss.py
import numpy as np
import pickle
from pathlib import Path
from food2recipe.core.settings import load_settings
from food2recipe.core.logging_utils import setup_logger

logger = setup_logger("index_faiss")

class RetrievalIndex:
    def __init__(self, settings=None):
        self.settings = settings or load_settings()
        self.index = None
        self.metadata = []
        self.is_faiss = False
        
    def build(self, embeddings: np.ndarray, metadata: list):
        """
        Builds the index.
        embeddings: (N, D) numpy array, normalized.
        """
        d = embeddings.shape[1]
        self.metadata = metadata
        
        if self.settings.USE_FAISS:
            try:
                import faiss
                # Inner Product (Cosine sim if normalized)
                self.index = faiss.IndexFlatIP(d)
                self.index.add(embeddings)
                self.is_faiss = True
                logger.info(f"Built FAISS index with {len(embeddings)} vectors.")
            except ImportError:
                logger.warning("FAISS not found. Fallback to Numpy.")
                self.index = embeddings
                self.is_faiss = False
        else:
            self.index = embeddings
            self.is_faiss = False
            logger.info(f"Built Numpy 'index' (brute force) with {len(embeddings)} vectors.")

    def search(self, query_emb: np.ndarray, k: int = 5):
        """
        query_emb: (1, D)
        Returns: distances, indices
        """
        if self.is_faiss:
            # FAISS expects float32
            if query_emb.dtype != np.float32:
                query_emb = query_emb.astype(np.float32)
            D, I = self.index.search(query_emb, k)
            return D[0], I[0] # Return 1D lists
        else:
            # Numpy Brute Force
            # Cosine similarity = dot product if normalized
            scores = np.dot(self.index, query_emb.T).flatten() # (N,)
            # Top-k
            # argsort returns lowest to highest, so we take last k and reverse
            indices = np.argsort(scores)[-k:][::-1]
            distances = scores[indices]
            return distances, indices

    def save(self, folder: Path):
        folder.mkdir(parents=True, exist_ok=True)
        # Save metadata
        with open(folder / "metadata.pkl", "wb") as f:
            pickle.dump(self.metadata, f)
            
        # Save index
        if self.is_faiss:
            import faiss
            faiss.write_index(self.index, str(folder / "faiss_index.bin"))
        else:
            np.save(folder / "numpy_index.npy", self.index)
            
    def load(self, folder: Path):
        if not (folder / "metadata.pkl").exists():
            raise FileNotFoundError(f"Index not found at {folder}")
            
        with open(folder / "metadata.pkl", "rb") as f:
            self.metadata = pickle.load(f)
            
        if self.settings.USE_FAISS and (folder / "faiss_index.bin").exists():
            import faiss
            self.index = faiss.read_index(str(folder / "faiss_index.bin"))
            self.is_faiss = True
        elif (folder / "numpy_index.npy").exists():
            self.index = np.load(folder / "numpy_index.npy")
            self.is_faiss = False
        else:
            raise FileNotFoundError("Index binary not found.")
        
        logger.info(f"Index loaded. Size: {len(self.metadata)}")
