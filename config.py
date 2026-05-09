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
RISK_PROFILES_DIR = os.path.join(DATA_DIR, "risk_profiles")

# ============================================================
# Multi-Year Configuration
# ============================================================
AVAILABLE_YEARS = [2021, 2022, 2023, 2024, 2025]
DEFAULT_YEAR = 2025

def get_year_dir(base_dir: str, year: int) -> str:
    """Get the year-specific subdirectory path."""
    return os.path.join(base_dir, str(year))

def get_raw_dir(year: int) -> str:
    return get_year_dir(RAW_DIR, year)

def get_extracted_dir(year: int) -> str:
    return get_year_dir(EXTRACTED_DIR, year)

def get_chunks_dir(year: int) -> str:
    return get_year_dir(CHUNKS_DIR, year)

def get_embeddings_dir(year: int) -> str:
    return get_year_dir(EMBEDDINGS_DIR, year)

def get_risk_profiles_dir(year: int) -> str:
    return get_year_dir(RISK_PROFILES_DIR, year)

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
        "description": "Risks from supply chain disruptions, single-source supplier dependencies, raw material shortages, manufacturing concentration, and logistics bottlenecks. Look for mentions of specific suppliers, geographic concentration, component shortages, or inventory issues.",
        "query_templates": [
            "supply chain disruption single source supplier",
            "component shortage manufacturing dependency",
            "logistics bottleneck inventory risk",
        ],
    },
    {
        "id": "regulatory_legal",
        "name": "Regulatory / Legal Risk",
        "description": "Risks from government regulations, active legal proceedings, regulatory investigations, compliance failures, fines, and policy changes. Look for mentions of specific lawsuits, settlement amounts, regulatory agencies (FTC, SEC, DOJ), or new legislation.",
        "query_templates": [
            "government regulation compliance enforcement",
            "legal proceedings litigation settlement",
            "regulatory investigation penalty fine",
        ],
    },
    {
        "id": "competition",
        "name": "Competition Risk",
        "description": "Risks from market competition leading to pricing pressure, loss of market share, and competitive disadvantage. Look for mentions of specific competitors, margin compression, customer switching, or market entry barriers eroding.",
        "query_templates": [
            "competitive pressure market share loss",
            "pricing competition margin compression",
            "new market entrants competitive disadvantage",
        ],
    },
    {
        "id": "cybersecurity",
        "name": "Cybersecurity Risk",
        "description": "Risks from data breaches, cyberattacks, ransomware, system intrusions, and privacy violations. Look for mentions of past security incidents, breach notification costs, data protection regulations (GDPR, CCPA), or specific threat vectors.",
        "query_templates": [
            "data breach cybersecurity incident",
            "ransomware cyberattack system intrusion",
            "privacy violation data protection GDPR",
        ],
    },
    {
        "id": "demand_market",
        "name": "Demand / Market Risk",
        "description": "Risks from demand uncertainty, shifting consumer preferences, market saturation, product adoption failure, and revenue concentration. Look for customer concentration percentages, seasonal dependency, or declining product lines.",
        "query_templates": [
            "demand uncertainty consumer preference shift",
            "market saturation revenue concentration",
            "product adoption failure seasonal dependency",
        ],
    },
    {
        "id": "macroeconomic",
        "name": "Macroeconomic Risk",
        "description": "Risks from economic downturns, inflation, interest rate changes, currency fluctuations, and geopolitical instability. Look for quantified foreign exchange exposure, specific country risks, tariff impacts, or recession scenario analysis.",
        "query_templates": [
            "economic downturn recession GDP impact",
            "inflation interest rate currency fluctuation",
            "geopolitical instability tariff trade war",
        ],
    },
    {
        "id": "operational",
        "name": "Operational Risk",
        "description": "Risks from internal process failures, system outages, workforce challenges, quality control issues, and business continuity threats. Look for mentions of specific outage incidents, employee turnover rates, safety violations, or operational KPI impacts.",
        "query_templates": [
            "system outage operational failure",
            "workforce turnover talent retention",
            "quality control safety business continuity",
        ],
    },
    {
        "id": "ip_technology",
        "name": "IP / Technology Risk",
        "description": "Risks from intellectual property disputes, patent infringement claims, technology obsolescence, R&D failures, and AI/ML governance risks. Look for specific patent cases, IP litigation costs, technology migration challenges, or R&D write-offs.",
        "query_templates": [
            "patent infringement intellectual property dispute",
            "technology obsolescence platform migration",
            "R&D failure AI governance risk",
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
USE_API = True            # Set to False to use local Qwen model, True for Groq API
GROQ_MODEL = "llama-3.1-8b-instant"

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
