"""
Configuration and constants for the Risk Profiling RAG pipeline.
"""

import os

# ============================================================
# Project Paths
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
EXTRACTED_DIR = os.path.join(DATA_DIR, "extracted")
CHUNKS_DIR = os.path.join(DATA_DIR, "chunks")
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")

# ============================================================
# Target Companies (Ticker → Company Name)
# ============================================================
COMPANIES = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "TSLA": "Tesla, Inc.",
    "NVDA": "NVIDIA Corporation",
    "AMZN": "Amazon.com, Inc.",
    "AMD": "Advanced Micro Devices, Inc.",
}

# ============================================================
# SEC EDGAR Configuration
# ============================================================
SEC_EDGAR_BASE_URL = "https://efts.sec.gov/LATEST"
SEC_EDGAR_FILINGS_URL = "https://www.sec.gov/cgi-bin/browse-edgar"
SEC_EDGAR_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data"
SEC_USER_AGENT = "CS455Project research@university.edu"  # SEC requires a user-agent

# ============================================================
# Risk Taxonomy
# ============================================================
RISK_CATEGORIES = [
    {
        "id": "supply_chain",
        "name": "Supply Chain Risk",
        "description": "Risks related to supply chain dependencies, disruptions, single-source suppliers, and logistics.",
        "query_templates": [
            "supply chain disruption risk",
            "single source supplier dependency",
            "manufacturing and logistics risk",
        ],
    },
    {
        "id": "regulatory_legal",
        "name": "Regulatory / Legal Risk",
        "description": "Risks from government regulations, legal proceedings, compliance requirements, and policy changes.",
        "query_templates": [
            "government regulation compliance risk",
            "legal proceedings and litigation risk",
            "regulatory policy change impact",
        ],
    },
    {
        "id": "competition",
        "name": "Competition Risk",
        "description": "Risks from market competition, pricing pressure, loss of market share, and new entrants.",
        "query_templates": [
            "competitive pressure market share risk",
            "pricing competition and new entrants",
            "competitive landscape threats",
        ],
    },
    {
        "id": "cybersecurity",
        "name": "Cybersecurity Risk",
        "description": "Risks from data breaches, cyberattacks, information security, and privacy violations.",
        "query_templates": [
            "cybersecurity data breach risk",
            "information security and privacy risk",
            "cyberattack and system vulnerability",
        ],
    },
    {
        "id": "demand_market",
        "name": "Demand / Market Risk",
        "description": "Risks from demand uncertainty, changing consumer preferences, market volatility, and seasonality.",
        "query_templates": [
            "demand uncertainty consumer preference risk",
            "market volatility and revenue fluctuation",
            "seasonal demand and product adoption risk",
        ],
    },
    {
        "id": "macroeconomic",
        "name": "Macroeconomic Risk",
        "description": "Risks from economic downturns, inflation, interest rates, currency fluctuations, and geopolitical events.",
        "query_templates": [
            "macroeconomic downturn recession risk",
            "inflation interest rate currency risk",
            "geopolitical instability and trade policy",
        ],
    },
    {
        "id": "operational",
        "name": "Operational Risk",
        "description": "Risks from internal operations, workforce, infrastructure failures, and execution challenges.",
        "query_templates": [
            "operational failure infrastructure risk",
            "workforce talent retention risk",
            "business continuity and execution risk",
        ],
    },
    {
        "id": "ip_technology",
        "name": "IP / Technology Risk",
        "description": "Risks from intellectual property disputes, technology obsolescence, and R&D failures.",
        "query_templates": [
            "intellectual property patent risk",
            "technology obsolescence and innovation risk",
            "research and development failure risk",
        ],
    },
]

# ============================================================
# Chunking Configuration
# ============================================================
CHUNK_SIZE = 512          # tokens per chunk
CHUNK_OVERLAP = 100       # overlap tokens between consecutive chunks
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# ============================================================
# Embedding Configuration
# ============================================================
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSION = 384  # bge-small-en-v1.5 output dimension

# ============================================================
# Retrieval Configuration
# ============================================================
TOP_K = 5                 # number of chunks to retrieve per query
RETRIEVE_TOP_K = 20       # initial retrieval pool for reranking
RERANK_ENABLED = True     # enable cross-encoder reranking
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ============================================================
# LLM Configuration
# ============================================================
LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"
LLM_MAX_NEW_TOKENS = 512
LLM_TEMPERATURE = 0.1     # low temperature for structured output

# ============================================================
# Risk Profile JSON Schema
# ============================================================
RISK_PROFILE_SCHEMA = {
    "company": "string",
    "risk_category": "string",
    "is_present": "boolean",
    "severity": "low | medium | high",
    "explanation": "string (1-3 sentences)",
    "evidence_snippets": ["string"],
    "confidence": "float (0.0 - 1.0)",
}

# ============================================================
# Evaluation Configuration
# ============================================================
EVAL_RECALL_K = 5
EVAL_NDCG_K = 5
FAITHFULNESS_RUBRIC = {
    0: "Unsupported or hallucinated",
    1: "Partially supported by retrieved evidence",
    2: "Fully supported by retrieved evidence",
}
