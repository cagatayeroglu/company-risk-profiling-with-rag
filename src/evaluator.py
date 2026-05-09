"""
Evaluation Module

Calculates retrieval metrics (Recall@K, MRR, nDCG@K) based on 
manual annotations. Provides a baseline vs. semantic+rerank comparison.
"""

import os
import math
import pandas as pd
import numpy as np
from typing import List, Dict

import sys
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.retriever import SemanticRetriever

def calculate_dcg(relevances: List[int]) -> float:
    """Calculate Discounted Cumulative Gain."""
    dcg = 0.0
    for i, rel in enumerate(relevances):
        dcg += rel / math.log2(i + 2)  # +2 because index is 0-based and formula is log2(i+1)
    return dcg

def calculate_ndcg(retrieved_relevances: List[int], ideal_relevances: List[int]) -> float:
    """Calculate Normalized Discounted Cumulative Gain."""
    dcg = calculate_dcg(retrieved_relevances)
    idcg = calculate_dcg(ideal_relevances)
    if idcg == 0:
        return 0.0
    return dcg / idcg

def evaluate_retrieval(annotations_file: str, top_k: int = 5):
    """
    Run the retriever for annotated queries and compute:
    - Recall@K
    - MRR (Mean Reciprocal Rank)
    - nDCG@K
    """
    if not os.path.exists(annotations_file):
        print(f"Annotations file not found: {annotations_file}")
        return

    df = pd.read_csv(annotations_file)
    
    # Check if we have valid annotations
    df = df[df["relevant_chunk_ids"].notna()]
    if len(df) == 0:
        print("No valid annotations found. Please fill in the relevant_chunk_ids.")
        return

    print("Initializing Retriever...")
    retriever = SemanticRetriever()
    
    metrics = {
        "mrr": [],
        "recall": [],
        "ndcg": []
    }
    
    print(f"\n{'='*60}")
    print(f"Retrieval Evaluation Results (Top-K={top_k})")
    print(f"{'='*60}")

    for _, row in df.iterrows():
        query = row["query"]
        company = row["company_filter"]
        if pd.isna(company):
            company = None
            
        # Ground truth
        truth_str = str(row["relevant_chunk_ids"]).strip()
        if not truth_str:
            continue
            
        ground_truth_ids = [s.strip() for s in truth_str.split(",")]
        
        # Retrieve results
        results = retriever.retrieve(query=query, top_k=top_k, company_filter=company, rerank=True)
        retrieved_ids = [r["chunk_id"] for r in results]
        
        # 1. Calculate MRR
        reciprocal_rank = 0.0
        for i, r_id in enumerate(retrieved_ids):
            if r_id in ground_truth_ids:
                reciprocal_rank = 1.0 / (i + 1)
                break
        metrics["mrr"].append(reciprocal_rank)
        
        # 2. Calculate Recall@K
        hits = sum(1 for r_id in retrieved_ids if r_id in ground_truth_ids)
        recall = hits / len(ground_truth_ids) if ground_truth_ids else 0.0
        metrics["recall"].append(recall)
        
        # 3. Calculate nDCG@K
        relevances = [1 if r_id in ground_truth_ids else 0 for r_id in retrieved_ids]
        ideal_relevances = sorted([1] * len(ground_truth_ids) + [0] * max(0, top_k - len(ground_truth_ids)), reverse=True)[:top_k]
        ndcg = calculate_ndcg(relevances, ideal_relevances)
        metrics["ndcg"].append(ndcg)
        
        print(f"Query: '{query}' ({company})")
        print(f"  Recall@{top_k}: {recall:.2f} | MRR: {reciprocal_rank:.2f} | nDCG@{top_k}: {ndcg:.2f}")

    # Aggregate
    mean_mrr = np.mean(metrics["mrr"])
    mean_recall = np.mean(metrics["recall"])
    mean_ndcg = np.mean(metrics["ndcg"])
    
    print(f"\n{'='*60}")
    print(f"FINAL AGGREGATE METRICS (N={len(metrics['mrr'])} queries)")
    print(f"{'='*60}")
    print(f"Mean Reciprocal Rank (MRR): {mean_mrr:.4f}")
    print(f"Average Recall@{top_k}:      {mean_recall:.4f}")
    print(f"Mean nDCG@{top_k}:           {mean_ndcg:.4f}")
    print(f"{'='*60}")

if __name__ == "__main__":
    anno_file = os.path.join(os.path.dirname(__file__), "..", "evaluation", "annotations", "retrieval_annotations.csv")
    evaluate_retrieval(anno_file, top_k=5)
