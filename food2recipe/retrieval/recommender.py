# File: food2recipe/retrieval/recommender.py
import torch
import numpy as np
from collections import Counter
from food2recipe.core.settings import load_settings
from food2recipe.core.logging_utils import setup_logger
from food2recipe.models.image_encoder import ImageEncoder
from food2recipe.retrieval.index_faiss import RetrievalIndex
from food2recipe.preprocessing.text_preprocess import RecipeProcessor
from food2recipe.preprocessing.image_preprocess import get_transforms, load_and_transform_image

logger = setup_logger("recommender")

class RecipeRecommender:
    def __init__(self, settings=None):
        self.settings = settings or load_settings()
        self.encoder = None
        self.index = None
        self.recipe_processor = None
        self.related_engine = None
        self.transform = get_transforms(mode="inference", image_size=self.settings.IMAGE_SIZE)
        
    def load_resources(self):
        """Loads model, index, and recipes CSV."""
        logger.info("Loading resources...")
        
        # 1. Encoder
        self.encoder = ImageEncoder(self.settings)
        
        # 2. Index
        index_path = self.settings.ARTIFACTS_DIR / "index"
        self.index = RetrievalIndex(self.settings)
        try:
            self.index.load(index_path)
        except Exception as e:
            logger.error(f"Failed to load index: {e}. Did you run build_index.py?")
            raise
            
        # 3. Recipes
        self.recipe_processor = RecipeProcessor(self.settings)
        self.recipe_processor.load_and_process()

        # 4. Related Engine
        from food2recipe.retrieval.related_engine import RelatedEngine
        self.related_engine = RelatedEngine(self.settings)
        try:
            self.related_engine.load_resources()
        except Exception as e:
            logger.warning(f"Could not load related engine resources: {e}")
            self.related_engine = None
        
    def predict(self, image_file):
        """
        Returns:
            - best_food_name (str)
            - confidence (float)
            - recipe (dict) or None
            - top_k_items (list of dicts with name, score, image_path from train)
        """
        # 1. Encode
        img_tensor = load_and_transform_image(image_file, self.transform)
        # Add batch dim
        img_tensor = img_tensor.unsqueeze(0) 
        
        emb = self.encoder.encode(img_tensor).numpy() # (1, D)
        
        # 2. Search
        top_k = self.settings.TOP_K
        scores, indices = self.index.search(emb, k=top_k)
        
        # 3. Aggregate results
        # We retrieved K nearest training images. They have labels (food_name).
        retrieved_items = []
        votes = []
        
        for score, idx in zip(scores, indices):
            meta = self.index.metadata[idx]
            food_class = meta['food_name']
            
            # Populate info
            item = {
                "food_name": food_class,
                "score": float(score),
                "image_path": meta.get("image_path")
            }
            retrieved_items.append(item)
            votes.append(food_class)
            
        # Majority vote
        # Or Sum-score voting (better for embeddings)
        score_map = {}
        for item in retrieved_items:
            fname = item["food_name"]
            score_map[fname] = score_map.get(fname, 0.0) + item["score"]
            
        # Sort by total score
        sorted_preds = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
        best_food_name, total_score = sorted_preds[0]
        
        # Start Confidence calculation
        # Simple heuristic: max(score) of top-1 match
        # Or avg score of retrieval
        top_1_score = retrieved_items[0]["score"] # The single nearest neighbor similarity
        
        # Create Deduplicated Top-K List
        dedup_topk = []
        for name, score in sorted_preds:
             dedup_topk.append({
                 "food_name": name,
                 "score": float(score / top_k), # Average score? Or just Sum? Sum is fine for ranking. 
                 # Let's keep it normalized roughly to 0-1 range for UI display if possible, 
                 # but similarity sums can be > 1. 
                 # Let's use avg matching score as a proxy:
                 "score": float(score) / top_k if top_k > 0 else 0.0,
                 # We lose specific image path here, but that's okay for class-level prediction
                 "image_path": None 
             })
             
        # Improve Confidence: use the aggregated score of the winner vs runner up?
        # stick to top-1 NN for "Confidence" as it represents "is there an identical image?"
        
        
        # Output Decision
        threshold = self.settings.CONFIDENCE_THRESHOLD
        
        is_uncertain = top_1_score < threshold
        

        # Get Recipe
        recipe = self.recipe_processor.get_recipe(best_food_name)
        
        # --- NEW: Get Related & Group Items ---
        related_similar = []
        related_group = []
        group_name = ""
        
        if self.related_engine:
            related_similar = self.related_engine.get_similar_dishes(best_food_name)
            related_group = self.related_engine.get_group_dishes(best_food_name)
            group_name = self.related_engine.get_group_name(best_food_name)

        result = {
            "predicted_food": best_food_name,
            "confidence": top_1_score,
            "is_uncertain": is_uncertain,
            "recipe": recipe,
            "top_k_items": dedup_topk,
            # New fields
            "related_similar": related_similar,
            "related_group": related_group,
            "group_name": group_name
        }
        
        return result

if __name__ == "__main__":
    # Smoke test
    pass
