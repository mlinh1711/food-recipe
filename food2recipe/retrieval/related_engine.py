
import numpy as np
from typing import List, Dict, Tuple
from food2recipe.core.settings import load_settings
from food2recipe.core.logging_utils import setup_logger

logger = setup_logger("related_engine")

# Hardcoded Groups for 30 Dishes
DISH_GROUPS = {
    # Cakes / Dumplings
    "Banh beo": "Bánh & Đồ Ăn Vặt",
    "Banh bot loc": "Bánh & Đồ Ăn Vặt",
    "Banh can": "Bánh & Đồ Ăn Vặt",
    "Banh chung": "Bánh & Đồ Ăn Vặt",
    "Banh cuon": "Bánh & Đồ Ăn Vặt", 
    "Banh duc": "Bánh & Đồ Ăn Vặt",
    "Banh gio": "Bánh & Đồ Ăn Vặt",
    "Banh khot": "Bánh & Đồ Ăn Vặt",
    "Banh mi": "Bánh & Đồ Ăn Vặt", # Arguable but fits
    "Banh pia": "Bánh & Đồ Ăn Vặt",
    "Banh tet": "Bánh & Đồ Ăn Vặt",
    "Banh trang nuong": "Bánh & Đồ Ăn Vặt",
    "Banh xeo": "Bánh & Đồ Ăn Vặt",
    "Nem chua": "Bánh & Đồ Ăn Vặt",
    "Goi cuon": "Bánh & Đồ Ăn Vặt",
    "Canh chua": "Món Chính & Cơm", # Soup but eaten with rice
    "Ca kho to": "Món Chính & Cơm",
    "Com tam": "Món Chính & Cơm",
    "Chao long": "Cháo & Súp",
    "Xoi xeo": "Món Chính & Cơm",
    # Noodles
    "Banh canh": "Bún & Mì",
    "Bun bo Hue": "Bún & Mì",
    "Bun dau mam tom": "Bún & Mì",
    "Bun mam": "Bún & Mì",
    "Bun rieu": "Bún & Mì",
    "Bun thit nuong": "Bún & Mì",
    "Cao lau": "Bún & Mì",
    "Hu tieu": "Bún & Mì",
    "Mi quang": "Bún & Mì",
    "Pho": "Bún & Mì",
}

class RelatedEngine:
    def __init__(self, settings=None):
        self.settings = settings or load_settings()
        self.centroids = {}
        self.groups = DISH_GROUPS
        
    def load_resources(self):
        """Loads class centroids."""
        centroids_path = self.settings.ARTIFACTS_DIR / "class_centroids.npy"
        if list(centroids_path.parent.glob("class_centroids.npy")):
            try:
                self.centroids = np.load(centroids_path, allow_pickle=True).item()
                logger.info(f"Loaded centroids for {len(self.centroids)} classes.")
            except Exception as e:
                logger.warning(f"Failed to load centroids: {e}. 'Similar' feature will be unavailable.")
        else:
            logger.warning("Centroids file not found. 'Similar' feature will be unavailable.")

    def get_similar_dishes(self, current_dish: str, k=3) -> List[str]:
        """
        Returns k similar dishes based on embedding distance.
        """
        if not self.centroids or current_dish not in self.centroids:
            return []
            
        query_vec = self.centroids[current_dish]
        scores = []
        
        for other, vec in self.centroids.items():
            if other == current_dish:
                continue
            # Cosine sim (vectors are normalized)
            sim = np.dot(query_vec, vec)
            scores.append((other, sim))
            
        scores.sort(key=lambda x: x[1], reverse=True)
        return [x[0] for x in scores[:k]]

    def get_group_dishes(self, current_dish: str, k=5) -> List[str]:
        """
        Returns dishes in the same group.
        """
        group = self.groups.get(current_dish)
        if not group:
            return []
            
        same_group = [d for d, g in self.groups.items() if g == group and d != current_dish]
        # Shuffle or stable sort? Stable sort by name for now, or random?
        # Let's just return first k
        return same_group[:k]
        
    def get_group_name(self, dish_name: str) -> str:
        return self.groups.get(dish_name, "Khác")

class SessionManager:
    """
    Manages session-based re-ranking.
    """
    def __init__(self):
        # liked_dishes: Set[str] = {"Pho", "Ban bo hue"}
        # disliked_dishes: Set[str] = ...
        pass
        
    @staticmethod
    def re_rank(top_k_items: List[Dict], liked: set, disliked: set) -> List[Dict]:
        """
        Re-ranks top_k_items based on session feedback.
        Simple heuristic:
        - Liked items: boost score * 1.2 or +0.1
        - Disliked items: penalty score * 0.5 or -0.1
        - Can also boost items in same group as liked items? (Advanced)
        """
        if not liked and not disliked:
            return top_k_items
            
        reranked = []
        for item in top_k_items:
            # Clone to avoid mutating original
            new_item = item.copy()
            name = item["food_name"]
            
            # Base Score
            score = item["score"]
            
            # Apply feedback
            if name in liked:
                score += 0.15 # Large boost
            elif name in disliked:
                score -= 0.15 # Large penalty
                
            new_item["original_score"] = item["score"]
            new_item["score"] = score
            reranked.append(new_item)
            
        # Re-sort
        reranked.sort(key=lambda x: x["score"], reverse=True)
        return reranked

