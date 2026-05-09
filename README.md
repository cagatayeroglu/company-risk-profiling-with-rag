# Automated Risk Profiling of Public Companies Using Multi-Document RAG

**CS 455 / CS 555 — Large Language Models — Spring 2025/2026**

## Overview

An LLM-based financial document intelligence system that automatically extracts, structures, and compares company-level risk profiles from SEC Form 10-K annual reports using a multi-document Retrieval-Augmented Generation (RAG) pipeline.

## Pipeline Architecture

```
10-K Collection → Section Extraction → Chunking + Metadata → Embedding + FAISS Index
                                                                        ↓
Dashboard + Comparison ← Risk Profile Aggregation ← LLM Structured Extraction ← Retrieval
```

## Target Companies

| Ticker | Company |
|--------|---------|
| AAPL | Apple Inc. |
| MSFT | Microsoft Corporation |
| TSLA | Tesla, Inc. |
| NVDA | NVIDIA Corporation |
| AMZN | Amazon.com, Inc. |
| AMD | Advanced Micro Devices, Inc. |

## Risk Taxonomy

1. Supply Chain Risk
2. Regulatory / Legal Risk
3. Competition Risk
4. Cybersecurity Risk
5. Demand / Market Risk
6. Macroeconomic Risk
7. Operational Risk
8. IP / Technology Risk

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Step 1: Collect 10-K reports from SEC EDGAR
python src/collector.py

# Step 2: Extract Item 1A Risk Factors
python src/extractor.py

# Step 3: Chunk and embed (Week 2)
python src/chunker.py
python src/embedder.py

# Step 4: Run risk extraction (Week 3)
python src/risk_extractor.py

# Step 5: Launch dashboard (Week 4)
streamlit run app.py
```

## Project Structure

```
CS455-PROJECT/
├── config.py                    # Configuration and constants
├── requirements.txt             # Python dependencies
├── data/
│   ├── raw/                     # Raw 10-K filings (HTML)
│   ├── extracted/               # Extracted Item 1A text
│   ├── chunks/                  # Chunked text with metadata
│   └── embeddings/              # FAISS index files
├── src/
│   ├── collector.py             # SEC EDGAR data collection
│   ├── extractor.py             # Item 1A section extraction
│   ├── chunker.py               # Text chunking + metadata
│   ├── embedder.py              # Embedding + FAISS indexing
│   ├── retriever.py             # Semantic retrieval
│   ├── risk_extractor.py        # LLM-based risk extraction
│   ├── comparator.py            # Company comparison
│   └── evaluator.py             # Evaluation metrics
├── prompts/
│   └── risk_extraction.py       # LLM prompt templates
├── evaluation/                  # Evaluation data and results
├── app.py                       # Streamlit dashboard
└── notebooks/                   # Exploration notebooks
```

## References

- [SEC EDGAR API](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)
- [FAISS](https://faiss.ai/index.html)
- [Sentence-Transformers](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.*
