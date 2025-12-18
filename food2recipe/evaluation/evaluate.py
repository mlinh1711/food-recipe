# File: food2recipe/evaluation/evaluate.py
import pandas as pd
from tqdm import tqdm
from food2recipe.core.settings import load_settings
from food2recipe.core.logging_utils import setup_logger
from food2recipe.retrieval.recommender import RecipeRecommender
from food2recipe.evaluation.metrics import compute_top_k_accuracy, compute_top_k_hit_rate, compute_mrr
from food2recipe.evaluation.report import save_report

logger = setup_logger("evaluate")

def run_evaluation():
    settings = load_settings()
    
    # Load Recommender
    recommender = RecipeRecommender(settings)
    try:
        recommender.load_resources()
    except Exception:
        logger.error("Could not load resources. Is the index built?")
        return

    # Load Test Data
    manifest_path = settings.ARTIFACTS_DIR / "manifest.csv"
    if not manifest_path.exists():
        logger.error("Manifest not found.")
        return
        
    df = pd.read_csv(manifest_path)
    test_df = df[df['split'] == 'test']
    
    if test_df.empty:
        logger.warning("No test data found in manifest. Using 'val' set for demo purposes?")
        # Fallback for user convenience if they haven't split data yet
        test_df = df[df['split'] == 'val']
        if test_df.empty:
             logger.error("No test or val data found.")
             return

    logger.info(f"Evaluating on {len(test_df)} images.")
    
    true_labels = []
    pred_labels_top1 = []
    pred_labels_topk = []
    
    details = []
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        img_path = row['image_path']
        true_food = row['food_name']
        
        try:
            # Predict
            # Note: predict() takes path or file
            res = recommender.predict(img_path)
            
            p_top1 = res['predicted_food']
            p_topk = [item['food_name'] for item in res['top_k_items']]
            
            true_labels.append(true_food)
            pred_labels_top1.append(p_top1)
            pred_labels_topk.append(p_topk)
            
            details.append({
                "image_path": img_path,
                "true_label": true_food,
                "predicted_label": p_top1,
                "is_correct": p_top1 == true_food,
                "confidence": res['confidence'],
                "top_k": str(p_topk)
            })
            
        except Exception as e:
            logger.error(f"Error eval image {img_path}: {e}")
            
    # Metrics
    acc_top1 = compute_top_k_accuracy(pred_labels_top1, true_labels)
    acc_top5 = compute_top_k_hit_rate(pred_labels_topk, true_labels) # checks if true in top-k (default 5 in settings)
    mrr = compute_mrr(pred_labels_topk, true_labels)
    
    metrics = {
        "Top-1 Accuracy": acc_top1,
        "Top-K Accuracy": acc_top5,
        "MRR": mrr
    }
    
    logger.info(f"Results: {metrics}")
    
    # Save Report
    s_path, c_path = save_report(metrics, details)
    logger.info(f"Reports saved to {s_path} and {c_path}")

if __name__ == "__main__":
    run_evaluation()
