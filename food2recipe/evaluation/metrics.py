# File: food2recipe/evaluation/metrics.py
import numpy as np
from typing import List

def compute_top_k_accuracy(predictions: List[str], ground_truth: List[str]):
    """
    Exact match accuracy (Top-1).
    """
    correct = 0
    for p, g in zip(predictions, ground_truth):
        if p == g:
            correct += 1
    return correct / len(ground_truth) if ground_truth else 0.0

def compute_top_k_hit_rate(top_k_predictions: List[List[str]], ground_truth: List[str]):
    """
    Checks if ground_truth is ANYWHERE in the top_k predictions list.
    """
    hits = 0
    for preds, g in zip(top_k_predictions, ground_truth):
        if g in preds:
            hits += 1
    return hits / len(ground_truth) if ground_truth else 0.0

def compute_mrr(top_k_predictions: List[List[str]], ground_truth: List[str]):
    """
    Mean Reciprocal Rank.
    """
    reciprocal_ranks = []
    for preds, g in zip(top_k_predictions, ground_truth):
        if g in preds:
            rank = preds.index(g) + 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
