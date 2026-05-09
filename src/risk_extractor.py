"""
LLM-based Structured Risk Extractor

Uses Qwen2.5-3B-Instruct to extract structured risk profiles
from retrieved evidence chunks. Supports both local and Colab execution.
"""

import os
import re
import json
from typing import Any

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    COMPANIES, RISK_CATEGORIES, LLM_MODEL,
    LLM_MAX_NEW_TOKENS, LLM_TEMPERATURE, TOP_K,
    EMBEDDINGS_DIR,
)
from prompts.risk_extraction import build_prompt, RISK_PROFILE_JSON_SCHEMA


# ============================================================
# JSON Parsing & Validation
# ============================================================

def extract_json_from_response(response: str) -> dict | None:
    """
    Extract a JSON object from the LLM response text.
    Handles cases where the model includes extra text around the JSON.
    """
    # Try direct parse first
    response = response.strip()
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # Try to find JSON between curly braces
    json_match = re.search(r"\{[\s\S]*\}", response)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # Try to fix common JSON issues
    cleaned = response
    # Remove trailing commas before closing braces/brackets
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
    # Fix single quotes to double quotes
    cleaned = cleaned.replace("'", '"')

    json_match = re.search(r"\{[\s\S]*\}", cleaned)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    return None


def validate_risk_profile(profile: dict) -> dict:
    """
    Validate and fix a risk profile dict against the expected schema.
    Fills in defaults for missing fields.
    """
    required_fields = {
        "company": "",
        "risk_category": "",
        "is_present": False,
        "severity": "low",
        "explanation": "No evidence found for this risk category.",
        "evidence_snippets": [],
        "confidence": 0.0,
    }

    # Fill missing fields with defaults
    for field, default in required_fields.items():
        if field not in profile:
            profile[field] = default

    # Validate severity
    if profile["severity"] not in ("low", "medium", "high"):
        profile["severity"] = "low"

    # Validate confidence range
    if not isinstance(profile["confidence"], (int, float)):
        profile["confidence"] = 0.0
    profile["confidence"] = max(0.0, min(1.0, float(profile["confidence"])))

    # Ensure evidence_snippets is a list of strings
    if not isinstance(profile["evidence_snippets"], list):
        profile["evidence_snippets"] = []
    profile["evidence_snippets"] = [
        str(s) for s in profile["evidence_snippets"][:3]
    ]

    # Ensure is_present is boolean
    profile["is_present"] = bool(profile["is_present"])

    return profile


# ============================================================
# LLM Model Loading
# ============================================================

class RiskExtractor:
    """
    Extracts structured risk profiles using an LLM.
    Supports Qwen2.5-3B-Instruct via HuggingFace transformers.
    """

    def __init__(self, model_name: str = None, device: str = "auto"):
        """
        Initialize the risk extractor.

        Args:
            model_name: HuggingFace model name (default from config)
            device: Device to load model on ("auto", "cpu", "cuda", "mps")
        """
        self.model_name = model_name or LLM_MODEL
        self.device = device
        self.model = None
        self.tokenizer = None
        self.retriever = None

    def load_model(self):
        """Load the LLM model and tokenizer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]
        import torch  # type: ignore[import-not-found]

        print(f"Loading model: {self.model_name}")
        print(f"Device: {self.device}")

        # Determine device
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
            print(f"Auto-detected device: {self.device}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        # Load model with appropriate settings
        load_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        if self.device == "cuda":
            load_kwargs["torch_dtype"] = torch.float16
            load_kwargs["device_map"] = "auto"
        elif self.device == "mps":
            load_kwargs["torch_dtype"] = torch.float16
        else:
            load_kwargs["torch_dtype"] = torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **load_kwargs,
        )

        if self.device == "mps":
            self.model = self.model.to("mps")

        self.model.eval()
        print(f"Model loaded successfully on {self.device}")

    def load_retriever(self):
        """Load the semantic retriever."""
        from src.retriever import SemanticRetriever
        self.retriever = SemanticRetriever()

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generate a response from the LLM.

        Args:
            system_prompt: System instruction
            user_prompt: User query with evidence

        Returns:
            Generated text response
        """
        import torch  # type: ignore[import-not-found]

        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Build chat messages for Qwen format
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=LLM_MAX_NEW_TOKENS,
                temperature=LLM_TEMPERATURE,
                do_sample=True if LLM_TEMPERATURE > 0 else False,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        input_len = inputs["input_ids"].shape[1]
        response = self.tokenizer.decode(
            outputs[0][input_len:],
            skip_special_tokens=True,
        )

        return response.strip()

    def extract_risk_profile(
        self,
        ticker: str,
        risk_category: dict,
        evidence_chunks: list[dict] = None,
    ) -> dict:
        """
        Extract a structured risk profile for a company and risk category.

        Args:
            ticker: Company ticker symbol
            risk_category: Risk category dict from config
            evidence_chunks: Pre-retrieved evidence chunks (optional)

        Returns:
            Validated risk profile dict
        """
        company_name = COMPANIES.get(ticker, ticker)
        cat_name = risk_category["name"]
        cat_desc = risk_category["description"]

        # Retrieve evidence if not provided
        if evidence_chunks is None:
            if self.retriever is None:
                self.load_retriever()
            evidence_chunks = self.retriever.retrieve_for_risk_category(
                risk_category, company=ticker, top_k=TOP_K,
            )

        print(f"  [{ticker}] {cat_name}: {len(evidence_chunks)} evidence chunks")

        # Build prompt
        system_prompt, user_prompt = build_prompt(
            ticker=ticker,
            company_name=company_name,
            risk_category=cat_name,
            risk_description=cat_desc,
            evidence_chunks=evidence_chunks,
        )

        # Generate response
        response = self.generate(system_prompt, user_prompt)

        # Parse JSON
        profile = extract_json_from_response(response)

        if profile is None:
            print(f"    WARNING: Failed to parse JSON from LLM response")
            print(f"    Raw response: {response[:200]}...")
            profile = {
                "company": ticker,
                "risk_category": cat_name,
                "is_present": False,
                "severity": "low",
                "explanation": "LLM response could not be parsed.",
                "evidence_snippets": [],
                "confidence": 0.0,
            }

        # Validate and fix
        profile["company"] = ticker
        profile["risk_category"] = cat_name
        profile = validate_risk_profile(profile)

        return profile

    def extract_company_profile(self, ticker: str) -> dict:
        """
        Extract risk profiles for all categories for a single company.

        Returns:
            Dict with company info and list of risk assessments
        """
        company_name = COMPANIES.get(ticker, ticker)
        print(f"\n{'='*60}")
        print(f"Extracting risk profile: {company_name} ({ticker})")
        print(f"{'='*60}")

        risk_assessments = []
        for category in RISK_CATEGORIES:
            profile = self.extract_risk_profile(ticker, category)
            risk_assessments.append(profile)

            severity = profile["severity"].upper()
            conf = profile["confidence"]
            present = "✓" if profile["is_present"] else "✗"
            print(f"    {present} {category['name']}: {severity} (conf: {conf:.2f})")

        return {
            "company": ticker,
            "company_name": company_name,
            "risk_assessments": risk_assessments,
            "total_categories": len(RISK_CATEGORIES),
            "risks_found": sum(1 for r in risk_assessments if r["is_present"]),
        }

    def extract_all_profiles(self, output_dir: str = None) -> list[dict]:
        """
        Extract risk profiles for all companies.
        Saves results to JSON files.

        Returns:
            List of company profile dicts
        """
        if output_dir is None:
            output_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data", "risk_profiles",
            )
        os.makedirs(output_dir, exist_ok=True)

        all_profiles = []

        for ticker in COMPANIES:
            company_profile = self.extract_company_profile(ticker)
            all_profiles.append(company_profile)

            # Save per-company profile
            company_path = os.path.join(output_dir, f"{ticker}_risk_profile.json")
            with open(company_path, "w", encoding="utf-8") as f:
                json.dump(company_profile, f, indent=2, ensure_ascii=False)

        # Save combined profiles
        combined_path = os.path.join(output_dir, "all_risk_profiles.json")
        with open(combined_path, "w", encoding="utf-8") as f:
            json.dump(all_profiles, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*60}")
        print("Extraction Summary")
        print(f"{'='*60}")
        for p in all_profiles:
            print(f"  {p['company']}: {p['risks_found']}/{p['total_categories']} risks found")
        print(f"\nResults saved to: {output_dir}")

        return all_profiles


if __name__ == "__main__":
    print("=" * 60)
    print("LLM Risk Profile Extractor")
    print(f"Model: {LLM_MODEL}")
    print("=" * 60)

    extractor = RiskExtractor()
    extractor.load_model()
    extractor.extract_all_profiles()
