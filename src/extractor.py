"""
Item 1A Risk Factors Extractor

Extracts the "Item 1A. Risk Factors" section from 10-K HTML filings.
Handles different formatting styles across companies.
"""

import os
import re
import json
import glob
import warnings
from typing import Optional
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

# SEC EDGAR filings are HTML but lxml occasionally warns about XML structure
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import COMPANIES, RAW_DIR, EXTRACTED_DIR, get_raw_dir, get_extracted_dir


def load_html(filepath: str) -> str:
    """Load an HTML file and return its content."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def html_to_text(html_content: str) -> str:
    """
    Convert HTML to clean text using BeautifulSoup.
    Preserves paragraph structure while removing tags.
    """
    soup = BeautifulSoup(html_content, "lxml")

    # Remove script and style elements
    for element in soup(["script", "style", "meta", "link"]):
        element.decompose()

    # Get text with newlines between block elements
    text = soup.get_text(separator="\n")

    # Clean up whitespace
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if line:
            lines.append(line)

    return "\n".join(lines)


def extract_item_1a(text: str) -> Optional[str]:
    """
    Extract the Item 1A Risk Factors section from the full 10-K text.

    Strategy:
    1. Find the start of "Item 1A" section
    2. Find the start of the next section ("Item 1B" or "Item 2")
    3. Extract everything in between
    """
    # Patterns for the start of Item 1A
    start_patterns = [
        r"(?i)item\s+1a[\.\s\-–—:]*\s*risk\s+factors",
        r"(?i)ITEM\s+1A[\.\s\-–—:]*\s*RISK\s+FACTORS",
        r"(?i)item\s+1a\.\s+risk\s+factors",
    ]

    # Patterns for the end of Item 1A (start of next section)
    end_patterns = [
        r"(?i)item\s+1b[\.\s\-–—:]*\s*unresolved\s+staff\s+comments",
        r"(?i)item\s+1b[\.\s\-–—:]",
        r"(?i)item\s+2[\.\s\-–—:]*\s*properties",
        r"(?i)item\s+2[\.\s\-–—:]",
    ]

    # Find the start position
    start_pos = None
    for pattern in start_patterns:
        matches = list(re.finditer(pattern, text))
        if matches:
            # Use the LAST match — the first might be from the table of contents
            start_pos = matches[-1].start()
            break

    if start_pos is None:
        return None

    # Find the end position (search only after start_pos)
    end_pos = len(text)
    search_text = text[start_pos + 50:]  # Skip past the "Item 1A" heading itself

    for pattern in end_patterns:
        matches = list(re.finditer(pattern, search_text))
        if matches:
            # Use the first match after Item 1A
            candidate_end = start_pos + 50 + matches[0].start()
            if candidate_end < end_pos:
                end_pos = candidate_end
                break

    extracted = text[start_pos:end_pos].strip()

    return extracted if len(extracted) > 200 else None  # Sanity check: too short = failed


def clean_extracted_text(text: str) -> str:
    """
    Clean the extracted Item 1A text.
    - Remove excessive whitespace
    - Remove page numbers and headers/footers
    - Normalize unicode
    """
    # Remove common page-level artifacts
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)  # Standalone page numbers
    text = re.sub(r"Table of Contents", "", text, flags=re.IGNORECASE)

    # Normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)  # Max 2 consecutive newlines
    text = re.sub(r"[ \t]+", " ", text)  # Collapse spaces
    text = re.sub(r" \n", "\n", text)  # Trailing spaces before newline

    # Normalize quotes and dashes
    text = text.replace("\u2019", "'")
    text = text.replace("\u2018", "'")
    text = text.replace("\u201c", '"')
    text = text.replace("\u201d", '"')
    text = text.replace("\u2013", "-")
    text = text.replace("\u2014", " - ")

    return text.strip()


def process_single_file(filepath: str, ticker: str, company_name: str, year: int = None) -> Optional[dict]:
    """
    Process a single 10-K file: load HTML → convert to text → extract Item 1A → clean.
    Returns metadata dict with extraction results.
    """
    print(f"  Loading: {os.path.basename(filepath)}")

    # Load and convert
    html_content = load_html(filepath)
    full_text = html_to_text(html_content)
    print(f"  Full text length: {len(full_text):,} chars")

    # Extract Item 1A
    item_1a = extract_item_1a(full_text)
    if item_1a is None:
        print(f"  WARNING: Could not extract Item 1A for {ticker}")
        return None

    # Clean
    item_1a_clean = clean_extracted_text(item_1a)
    print(f"  Item 1A length: {len(item_1a_clean):,} chars")

    # Save extracted text to year-specific dir if year provided
    if year:
        out_dir = get_extracted_dir(year)
    else:
        out_dir = EXTRACTED_DIR
    os.makedirs(out_dir, exist_ok=True)
    output_filename = f"{ticker}_item1a.txt"
    output_path = os.path.join(out_dir, output_filename)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(item_1a_clean)

    print(f"  Saved to: {output_path}")

    return {
        "ticker": ticker,
        "company_name": company_name,
        "source_file": filepath,
        "output_file": output_path,
        "full_text_length": len(full_text),
        "item_1a_length": len(item_1a_clean),
        "extraction_ratio": round(len(item_1a_clean) / len(full_text) * 100, 2),
        "fiscal_year": year,
    }


def extract_all_item_1a() -> list:
    """
    Process all downloaded 10-K files and extract Item 1A sections.
    Returns a list of metadata dicts.
    """
    results = []

    # Load collection metadata if available
    meta_path = os.path.join(RAW_DIR, "collection_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            collection_meta = json.load(f)
    else:
        # Fallback: scan raw directory for files
        collection_meta = []
        for ticker, company_name in COMPANIES.items():
            pattern = os.path.join(RAW_DIR, f"{ticker}_10K_*")
            files = glob.glob(pattern)
            if files:
                collection_meta.append({
                    "ticker": ticker,
                    "company_name": company_name,
                    "local_path": files[0],
                })

    for filing in collection_meta:
        ticker = filing["ticker"]
        company_name = filing.get("company_name", COMPANIES.get(ticker, ticker))
        filepath = filing["local_path"]

        print(f"\n{'='*60}")
        print(f"Extracting Item 1A: {company_name} ({ticker})")
        print(f"{'='*60}")

        result = process_single_file(filepath, ticker, company_name)
        if result:
            results.append(result)

    # Save extraction metadata
    extraction_meta_path = os.path.join(EXTRACTED_DIR, "extraction_metadata.json")
    with open(extraction_meta_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nExtraction metadata saved to: {extraction_meta_path}")

    return results


def extract_all_item_1a_for_year(target_year: int) -> list:
    """
    Process all 10-K files for a specific fiscal year and extract Item 1A sections.
    """
    results = []
    year_raw_dir = get_raw_dir(target_year)
    
    # Load collection metadata for this year
    meta_path = os.path.join(year_raw_dir, "collection_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            collection_meta = json.load(f)
    else:
        # Fallback: scan year raw directory for files
        collection_meta = []
        for ticker, company_name in COMPANIES.items():
            pattern = os.path.join(year_raw_dir, f"{ticker}_10K_*")
            files = glob.glob(pattern)
            if files:
                collection_meta.append({
                    "ticker": ticker,
                    "company_name": company_name,
                    "local_path": files[0],
                    "fiscal_year": target_year,
                })

    for filing in collection_meta:
        ticker = filing["ticker"]
        company_name = filing.get("company_name", COMPANIES.get(ticker, ticker))
        filepath = filing["local_path"]

        print(f"\n{'='*60}")
        print(f"Extracting Item 1A: {company_name} ({ticker}) — FY{target_year}")
        print(f"{'='*60}")

        result = process_single_file(filepath, ticker, company_name, year=target_year)
        if result:
            results.append(result)

    # Save extraction metadata in year dir
    year_ext_dir = get_extracted_dir(target_year)
    os.makedirs(year_ext_dir, exist_ok=True)
    extraction_meta_path = os.path.join(year_ext_dir, "extraction_metadata.json")
    with open(extraction_meta_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nExtraction metadata saved to: {extraction_meta_path}")

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Item 1A Risk Factors Extractor")
    print("=" * 60)

    results = extract_all_item_1a()

    print(f"\n{'='*60}")
    print(f"Extraction Summary")
    print(f"{'='*60}")
    print(f"Total files processed: {len(results)}")
    for r in results:
        print(f"  {r['ticker']}: {r['item_1a_length']:,} chars "
              f"({r['extraction_ratio']}% of full text)")
