"""
Text Chunker with Metadata

Splits extracted Item 1A Risk Factors text into overlapping chunks
and attaches metadata (company, year, section, chunk_id).
"""

import os
import re
import json
import glob
from typing import Optional

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    COMPANIES, EXTRACTED_DIR, CHUNKS_DIR,
    CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_SEPARATORS,
    get_chunks_dir, get_extracted_dir,
)


def split_text_recursive(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    separators: list = None,
) -> list[str]:
    """
    Recursively split text into chunks using a hierarchy of separators.
    Similar to LangChain's RecursiveCharacterTextSplitter but standalone.

    Args:
        text: The text to split
        chunk_size: Maximum characters per chunk
        chunk_overlap: Number of overlapping characters between chunks
        separators: List of separators to try, in order of preference

    Returns:
        List of text chunks
    """
    if separators is None:
        separators = CHUNK_SEPARATORS

    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    # Find the best separator for this level
    chosen_sep = separators[-1]  # fallback to character-level
    for sep in separators:
        if sep in text:
            chosen_sep = sep
            break

    # Split by the chosen separator
    parts = text.split(chosen_sep)

    chunks = []
    current_chunk = ""

    for part in parts:
        candidate = current_chunk + chosen_sep + part if current_chunk else part

        if len(candidate) <= chunk_size:
            current_chunk = candidate
        else:
            # Save current chunk if it has content
            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            # If this part alone is too large, recurse with finer separators
            if len(part) > chunk_size and len(separators) > 1:
                sub_chunks = split_text_recursive(
                    part, chunk_size, chunk_overlap, separators[1:]
                )
                chunks.extend(sub_chunks)
                current_chunk = ""
            else:
                current_chunk = part

    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # Add overlap between consecutive chunks
    if chunk_overlap > 0 and len(chunks) > 1:
        overlapping_chunks = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapping_chunks.append(chunk)
            else:
                # Take the last `chunk_overlap` characters from previous chunk
                prev = chunks[i - 1]
                overlap_text = prev[-chunk_overlap:] if len(prev) > chunk_overlap else prev
                # Find a clean break point in the overlap
                clean_break = overlap_text.find(". ")
                if clean_break != -1:
                    overlap_text = overlap_text[clean_break + 2:]
                overlapping_chunks.append(overlap_text + " " + chunk)
            overlapping_chunks[-1] = overlapping_chunks[-1].strip()

        return overlapping_chunks

    return chunks


def extract_year_from_filename(filename: str) -> str:
    """Extract the fiscal year from the extracted text filename."""
    # Filenames like AAPL_item1a.txt — we'll check collection metadata
    return "2025"  # Will be overridden if metadata is available


def chunk_single_company(
    ticker: str,
    company_name: str,
    text_path: str,
    year: str = "2025",
) -> list[dict]:
    """
    Chunk a single company's Item 1A text and attach metadata.

    Returns:
        List of chunk dicts with metadata
    """
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"  Text length: {len(text):,} chars")

    # Split into chunks
    raw_chunks = split_text_recursive(text)
    print(f"  Generated {len(raw_chunks)} chunks")

    # Create chunk objects with metadata
    chunks = []
    for i, chunk_text in enumerate(raw_chunks):
        chunk = {
            "chunk_id": f"{ticker}_{year}_item1a_{i:04d}",
            "company": ticker,
            "company_name": company_name,
            "year": year,
            "section": "Item 1A - Risk Factors",
            "chunk_index": i,
            "total_chunks": len(raw_chunks),
            "text": chunk_text,
            "char_count": len(chunk_text),
        }
        chunks.append(chunk)

    return chunks


def chunk_all_companies() -> list[dict]:
    """
    Chunk all extracted Item 1A texts.
    Returns a combined list of all chunks with metadata.
    """
    all_chunks = []

    # Try to load collection metadata for filing dates
    filing_years = {}
    meta_paths = [
        os.path.join(EXTRACTED_DIR, "extraction_metadata.json"),
        os.path.join(os.path.dirname(EXTRACTED_DIR), "raw", "collection_metadata.json"),
    ]
    for meta_path in meta_paths:
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            for entry in meta:
                ticker = entry.get("ticker", "")
                filing_date = entry.get("filing_date", "")
                if filing_date:
                    filing_years[ticker] = filing_date[:4]

    # Process each extracted file
    for ticker, company_name in COMPANIES.items():
        text_path = os.path.join(EXTRACTED_DIR, f"{ticker}_item1a.txt")
        if not os.path.exists(text_path):
            print(f"  SKIP: {text_path} not found")
            continue

        year = filing_years.get(ticker, "2025")

        print(f"\n{'='*60}")
        print(f"Chunking: {company_name} ({ticker}) — Year: {year}")
        print(f"{'='*60}")

        chunks = chunk_single_company(ticker, company_name, text_path, year)
        all_chunks.extend(chunks)

    # Save all chunks
    os.makedirs(CHUNKS_DIR, exist_ok=True)

    # Save as a single JSON file
    chunks_path = os.path.join(CHUNKS_DIR, "all_chunks.json")
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    # Also save per-company files
    company_chunks = {}
    for chunk in all_chunks:
        ticker = chunk["company"]
        if ticker not in company_chunks:
            company_chunks[ticker] = []
        company_chunks[ticker].append(chunk)

    for ticker, chunks in company_chunks.items():
        company_path = os.path.join(CHUNKS_DIR, f"{ticker}_chunks.json")
        with open(company_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)

    # Save chunking metadata
    meta = {
        "total_chunks": len(all_chunks),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "companies": {
            ticker: len(chunks)
            for ticker, chunks in company_chunks.items()
        },
        "avg_chunk_length": round(
            sum(c["char_count"] for c in all_chunks) / len(all_chunks), 1
        ) if all_chunks else 0,
    }
    meta_path = os.path.join(CHUNKS_DIR, "chunking_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Chunking Summary")
    print(f"{'='*60}")
    print(f"Total chunks: {meta['total_chunks']}")
    print(f"Avg chunk length: {meta['avg_chunk_length']} chars")
    for ticker, count in meta["companies"].items():
        print(f"  {ticker}: {count} chunks")

    return all_chunks


def chunk_all_for_year(target_year: int) -> list[dict]:
    """
    Chunk all extracted Item 1A texts for a specific fiscal year.
    Returns a combined list of all chunks with metadata.
    """
    all_chunks = []
    year_ext_dir = get_extracted_dir(target_year)
    year_chunks_dir = get_chunks_dir(target_year)
    
    # Load filing metadata for this year
    filing_years = {}
    meta_path = os.path.join(year_ext_dir, "extraction_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        for entry in meta:
            filing_years[entry.get("ticker", "")] = str(target_year)

    # Process each extracted file
    for ticker, company_name in COMPANIES.items():
        text_path = os.path.join(year_ext_dir, f"{ticker}_item1a.txt")
        if not os.path.exists(text_path):
            print(f"  SKIP: {text_path} not found")
            continue

        year = str(target_year)

        print(f"\n{'='*60}")
        print(f"Chunking: {company_name} ({ticker}) — FY{year}")
        print(f"{'='*60}")

        chunks = chunk_single_company(ticker, company_name, text_path, year)
        all_chunks.extend(chunks)

    # Save all chunks to year-specific dir
    os.makedirs(year_chunks_dir, exist_ok=True)

    chunks_path = os.path.join(year_chunks_dir, "all_chunks.json")
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    # Save per-company files
    company_chunks = {}
    for chunk in all_chunks:
        ticker = chunk["company"]
        if ticker not in company_chunks:
            company_chunks[ticker] = []
        company_chunks[ticker].append(chunk)

    for ticker, chunks in company_chunks.items():
        company_path = os.path.join(year_chunks_dir, f"{ticker}_chunks.json")
        with open(company_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Chunking Summary (FY{target_year})")
    print(f"{'='*60}")
    print(f"Total chunks: {len(all_chunks)}")
    for ticker, chunks in company_chunks.items():
        print(f"  {ticker}: {len(chunks)} chunks")

    return all_chunks


if __name__ == "__main__":
    print("=" * 60)
    print("Text Chunker with Metadata")
    print("=" * 60)
    chunk_all_companies()
