"""
Risk Extraction Prompt Templates & JSON Schema

Defines the system prompt, user prompt templates, and the strict JSON schema
used by the LLM to produce structured risk profiles.
"""

# ============================================================
# JSON Output Schema
# ============================================================

RISK_PROFILE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "company": {"type": "string", "description": "Company ticker symbol"},
        "risk_category": {"type": "string", "description": "Risk category name"},
        "is_present": {"type": "boolean", "description": "Whether this risk is mentioned/present"},
        "severity": {
            "type": "string",
            "enum": ["low", "medium", "high"],
            "description": "Estimated severity level",
        },
        "explanation": {
            "type": "string",
            "description": "1-3 sentence explanation of the risk based on evidence",
        },
        "evidence_snippets": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Direct quotes from the source text supporting the assessment",
            "minItems": 1,
            "maxItems": 3,
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Confidence level in the assessment (0.0 to 1.0)",
        },
    },
    "required": [
        "company",
        "risk_category",
        "is_present",
        "severity",
        "explanation",
        "evidence_snippets",
        "confidence",
    ],
}

# ============================================================
# System Prompt
# ============================================================

SYSTEM_PROMPT = """You are a financial risk analyst AI. Your task is to analyze company risk disclosures from SEC Form 10-K filings and produce structured risk assessments.

RULES:
1. Base your assessment ONLY on the provided evidence text. Do NOT use external knowledge.
2. Output ONLY valid JSON matching the specified schema. No extra text before or after the JSON.
3. Every claim in your explanation MUST be directly supported by the evidence snippets.
4. If the evidence does not clearly mention the risk category, set is_present to false and severity to "low".
5. Evidence snippets must be direct quotes from the provided text, not paraphrased.
6. Keep explanations concise: 1-3 sentences maximum.
7. Confidence should reflect how clearly the evidence supports your assessment. Provide a precise, granular 2-decimal float (e.g., 0.94, 0.87, 0.73) rather than rounding to nearest tenths:
   - Near 0.95+: Very clear, explicit risk discussion in evidence
   - Near 0.80+: Moderate evidence, some interpretation needed
   - Near 0.60+: Weak or indirect evidence
   - Near 0.40+: Minimal evidence, mostly inferred
   - Near 0.10+: No relevant evidence found"""

# ============================================================
# User Prompt Template
# ============================================================

USER_PROMPT_TEMPLATE = """Analyze the following evidence chunks from {company_name} ({ticker})'s 10-K filing for the risk category: **{risk_category}**.

Risk Category Description: {risk_description}

--- EVIDENCE CHUNKS ---
{evidence_text}
--- END EVIDENCE ---

Based ONLY on the evidence above, produce a JSON risk assessment with this exact schema:
{{
    "company": "{ticker}",
    "risk_category": "{risk_category}",
    "is_present": true/false,
    "severity": "low" | "medium" | "high",
    "explanation": "1-3 sentence explanation based on evidence",
    "evidence_snippets": ["direct quote 1", "direct quote 2"],
    "confidence": 0.0 to 1.0
}}

Severity guidelines:
- "high": The company explicitly discusses significant, material risks in this category with potential major financial impact.
- "medium": The company mentions risks in this category but describes them as manageable or mitigated.
- "low": The risk is briefly mentioned or only indirectly relevant, or no clear evidence found.

Output ONLY the JSON object, nothing else:"""


def format_evidence_chunks(chunks: list[dict]) -> str:
    """
    Format retrieved evidence chunks into a string for the prompt.

    Args:
        chunks: List of chunk dicts with 'text', 'company', 'chunk_id' fields

    Returns:
        Formatted evidence string
    """
    evidence_parts = []
    for i, chunk in enumerate(chunks, 1):
        evidence_parts.append(
            f"[Chunk {i}] (ID: {chunk.get('chunk_id', 'unknown')})\n{chunk['text']}"
        )
    return "\n\n".join(evidence_parts)


def build_prompt(
    ticker: str,
    company_name: str,
    risk_category: str,
    risk_description: str,
    evidence_chunks: list[dict],
) -> tuple[str, str]:
    """
    Build the complete system + user prompt for risk extraction.

    Args:
        ticker: Company ticker symbol
        company_name: Full company name
        risk_category: Name of the risk category
        risk_description: Description of the risk category
        evidence_chunks: List of retrieved evidence chunk dicts

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    evidence_text = format_evidence_chunks(evidence_chunks)

    user_prompt = USER_PROMPT_TEMPLATE.format(
        company_name=company_name,
        ticker=ticker,
        risk_category=risk_category,
        risk_description=risk_description,
        evidence_text=evidence_text,
    )

    return SYSTEM_PROMPT, user_prompt
