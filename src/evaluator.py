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
from config import COMPANIES, RISK_CATEGORIES, DEFAULT_YEAR


def generate_annotation_scaffold(
    year: int = DEFAULT_YEAR,
    top_k: int = 10,
    out_path: str = None,
) -> str:
    """
    Build a FRESH labeling scaffold from the current index so gold labels match
    the current chunk_ids. For each risk category × company, retrieves the top
    candidates and writes a long-format CSV with an empty `is_relevant` column
    for you to fill (1 = relevant, 0/blank = not).

    Convert the filled file to evaluator format with `scaffold_to_annotations`.
    """
    retriever = SemanticRetriever(year=year)
    rows = []
    qid = 0
    for cat in RISK_CATEGORIES:
        query = cat["query_templates"][0]  # representative query
        for ticker in COMPANIES:
            qid += 1
            results = retriever.retrieve(query=query, top_k=top_k,
                                         company_filter=ticker, rerank=True)
            for r in results:
                rows.append({
                    "query_id": qid,
                    "query": query,
                    "company_filter": ticker,
                    "chunk_id": r["chunk_id"],
                    "relevance_score": round(r.get("relevance", 0.0), 3),
                    "is_relevant": "",  # <-- you fill: 1 or 0
                    "chunk_preview": r["text"][:160].replace("\n", " "),
                })
    if out_path is None:
        out_path = os.path.join(os.path.dirname(__file__), "..",
                                "evaluation", "annotations",
                                f"retrieval_scaffold_{year}.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Scaffold written: {out_path}  ({len(rows)} candidate rows, {qid} queries)")
    print("Fill the `is_relevant` column (1=relevant), then run scaffold_to_annotations().")
    return out_path


def scaffold_to_annotations(scaffold_path: str, out_path: str = None) -> str:
    """Aggregate a filled long-format scaffold into the wide annotation format
    (query, company_filter, relevant_chunk_ids) that evaluate_retrieval expects."""
    df = pd.read_csv(scaffold_path)
    df = df[df["is_relevant"].astype(str).str.strip().isin(["1", "1.0"])]
    grouped = (
        df.groupby(["query_id", "query", "company_filter"])["chunk_id"]
        .apply(lambda s: ", ".join(s)).reset_index()
        .rename(columns={"chunk_id": "relevant_chunk_ids"})
    )
    if out_path is None:
        out_path = scaffold_path.replace("scaffold", "annotations")
    grouped.to_csv(out_path, index=False)
    print(f"Annotations written: {out_path}  ({len(grouped)} labeled queries)")
    return out_path

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

def _score_run(retriever, df, top_k):
    """Run retrieval over the annotated queries and return mean MRR/Recall/nDCG."""
    mrr, recall, ndcg = [], [], []
    for _, row in df.iterrows():
        query = row["query"]
        company = None if pd.isna(row["company_filter"]) else row["company_filter"]
        truth = str(row["relevant_chunk_ids"]).strip()
        if not truth:
            continue
        gt = [s.strip() for s in truth.split(",")]

        results = retriever.retrieve(query=query, top_k=top_k,
                                     company_filter=company, rerank=True)
        rids = [r["chunk_id"] for r in results]

        rr = next((1.0 / (i + 1) for i, rid in enumerate(rids) if rid in gt), 0.0)
        mrr.append(rr)
        recall.append(sum(1 for rid in rids if rid in gt) / len(gt) if gt else 0.0)
        rels = [1 if rid in gt else 0 for rid in rids]
        ideal = sorted([1] * len(gt) + [0] * max(0, top_k - len(gt)), reverse=True)[:top_k]
        ndcg.append(calculate_ndcg(rels, ideal))
    return np.mean(mrr), np.mean(recall), np.mean(ndcg), len(mrr)


def evaluate_retrieval(annotations_file: str, top_k: int = 5, year: int = DEFAULT_YEAR,
                       ablation: bool = True):
    """
    Compute Recall@K, MRR, nDCG@K over annotated queries against the FY`year`
    index. When `ablation` is on, runs both HYBRID (BM25+dense) and DENSE-only
    to quantify the BM25 contribution. (Dense-only is simulated by disabling the
    retriever's BM25 index — same code path, no result-changing hacks.)
    """
    if not os.path.exists(annotations_file):
        print(f"Annotations file not found: {annotations_file}")
        print("Tip: generate_annotation_scaffold() then fill is_relevant, "
              "then scaffold_to_annotations().")
        return

    df = pd.read_csv(annotations_file)
    df = df[df["relevant_chunk_ids"].notna()]
    if len(df) == 0:
        print("No valid annotations found. Fill in relevant_chunk_ids first.")
        return

    print(f"Initializing retriever (FY{year})...")
    retriever = SemanticRetriever(year=year)
    # Ensure BM25 exists so the hybrid-vs-dense ablation works even when
    # HYBRID_ENABLED is False in production.
    if ablation and retriever.bm25 is None:
        retriever._build_bm25()

    print(f"\n{'='*60}\nRetrieval Evaluation (Top-K={top_k}, N={len(df)})\n{'='*60}")

    runs = [("HYBRID (BM25+dense)", retriever.bm25)]
    if ablation:
        runs.append(("DENSE-only", None))

    results = []
    saved_bm25 = retriever.bm25
    for label, bm25_state in runs:
        retriever.bm25 = bm25_state  # None => dense-only path
        mrr, rec, ndcg, n = _score_run(retriever, df, top_k)
        print(f"  {label:22s} | MRR={mrr:.4f}  Recall@{top_k}={rec:.4f}  nDCG@{top_k}={ndcg:.4f}")
        results.append({"method": label, "mrr": round(float(mrr), 4),
                        "recall": round(float(rec), 4), "ndcg": round(float(ndcg), 4)})
    retriever.bm25 = saved_bm25
    print(f"{'='*60}")
    return {"top_k": top_k, "n_queries": len(df), "runs": results}


def write_evaluation_report(year: int = DEFAULT_YEAR, top_k: int = 5) -> str:
    """Run quality + retrieval evaluation and write a human-readable Markdown
    report to evaluation/results/evaluation_report_<year>.md."""
    from src.quality_eval import evaluate_quality

    quality = evaluate_quality(year)
    anno = os.path.join(os.path.dirname(__file__), "..", "evaluation",
                        "annotations", f"retrieval_annotations_{year}.csv")
    retrieval = evaluate_retrieval(anno, top_k=top_k, year=year, ablation=True) \
        if os.path.exists(anno) else None

    out_dir = os.path.join(os.path.dirname(__file__), "..", "evaluation", "results")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"evaluation_report_{year}.md")

    from datetime import date
    L = [f"# Evaluation Report — FY{year}", f"_Generated: {date.today()}_", ""]

    if retrieval:
        L += ["## Retrieval Metrics",
              f"Labeled queries: {retrieval['n_queries']} | Top-K: {retrieval['top_k']}",
              "(Ground truth: LLM-labeled silver set — spot-check recommended.)", "",
              "| Method | MRR | Recall@K | nDCG@K |", "|---|---|---|---|"]
        for r in retrieval["runs"]:
            L.append(f"| {r['method']} | {r['mrr']:.4f} | {r['recall']:.4f} | {r['ndcg']:.4f} |")
        L.append("")

    if quality:
        sev = quality.get("severity_distribution", {})
        L += ["## Generation Quality",
              f"- **Grounding (faithfulness proxy):** {quality['grounded_snippets']}/"
              f"{quality['total_snippets']} snippets verbatim in source "
              f"(**{quality['grounding_pct']}%**)",
              f"- **Mean confidence:** {quality['mean_confidence']}",
              f"- **Mean evidence chunks:** {quality['mean_evidence_chunks']}",
              f"- **Extraction failures:** {quality['extraction_failures']}", "",
              "### Severity distribution",
              "| Severity | Count |", "|---|---|"]
        for lvl in ("negligible", "low", "medium", "high", "critical"):
            if sev.get(lvl):
                L.append(f"| {lvl} | {sev[lvl]} |")
        ung = quality.get("ungrounded_examples", [])
        if ung:
            L += ["", "### Ungrounded snippets (possible paraphrase/hallucination)"]
            for e in ung[:10]:
                L.append(f"- [{e['company']}] {e['category']}: \"{e['snippet']}…\"")
        L.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"\nReport written: {path}")
    return path


if __name__ == "__main__":
    yr = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_YEAR
    write_evaluation_report(yr, top_k=5)
