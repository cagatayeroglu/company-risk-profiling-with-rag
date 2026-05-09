"""
Risk Extraction Prompt Templates & JSON Schema

Defines the system prompt, user prompt templates, and the strict JSON schema
used by the LLM to produce structured risk profiles.
"""

from __future__ import annotations

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
            "enum": ["negligible", "low", "medium", "high", "critical"],
            "description": "Estimated severity level on 5-point scale",
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

SYSTEM_PROMPT = """You are a senior financial risk analyst specializing in SEC 10-K filings. You produce precise, differentiated risk assessments. You NEVER default to generic scores.

STRICT RULES:
1. Base your assessment ONLY on the provided evidence text. Do NOT use external knowledge.
2. Output ONLY valid JSON matching the specified schema. No extra text before or after the JSON.
3. Every claim in your explanation MUST be directly supported by the evidence snippets.
4. If the evidence does not clearly mention the risk category, set is_present to false and severity to "negligible".
5. Evidence snippets must be DIRECT QUOTES from the provided text — not paraphrased.
6. Keep explanations concise: 1-3 sentences maximum.

CRITICAL — CONFIDENCE SCORING:
You MUST vary your confidence scores meaningfully. Do NOT default to 0.94 or round numbers.
Think step-by-step about how strong the evidence actually is:
- 0.90-0.99: Multiple explicit, detailed paragraphs directly discussing this specific risk with quantified impact
- 0.75-0.89: Clear discussion of the risk but lacking specific financial figures or detailed mitigation plans
- 0.55-0.74: Risk is mentioned but only in passing, within broader context, or as boilerplate language
- 0.30-0.54: Only tangentially related evidence; requires significant interpretation
- 0.05-0.29: No meaningful evidence for this risk category

CRITICAL — SEVERITY DIFFERENTIATION:
You must carefully distinguish between severity levels. Most risks should NOT be "high" — reserve "high" and "critical" for truly exceptional risks with clear financial materiality."""

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
    "severity": "negligible" | "low" | "medium" | "high" | "critical",
    "explanation": "1-3 sentence explanation based on evidence",
    "evidence_snippets": ["direct quote 1", "direct quote 2"],
    "confidence": 0.0 to 1.0
}}

SEVERITY SCALE (use the full range — do NOT default to "high"):
- "critical": Company describes this as an existential or enterprise-threatening risk with potential for massive financial loss, regulatory shutdown, or fundamental business model disruption. Must include language like "material adverse effect", quantified losses, or active ongoing crises.
- "high": Company explicitly warns of significant, material risks with concrete examples of past incidents or high-probability future impacts. Evidence includes specific dollar amounts, lawsuits, or regulatory actions.
- "medium": Company acknowledges and discusses the risk meaningfully, describes mitigation strategies, but the risk is typical for the industry and well-managed. Standard risk factor language without exceptional urgency.
- "low": Risk is briefly mentioned in a general or boilerplate manner. No specific incidents, no quantified impact, listed among many generic risks without emphasis.
- "negligible": Risk is not meaningfully discussed. Only indirect or tangential references, or no relevant evidence at all.

IMPORTANT: Most well-managed companies have MEDIUM-level risks. Only assign "high" or "critical" if the evidence clearly justifies it with specific incidents, quantified impacts, or urgent language. Standard boilerplate risk factors = "medium" at most.

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
