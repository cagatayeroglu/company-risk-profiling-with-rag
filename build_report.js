const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, LevelFormat, HeadingLevel, BorderStyle, WidthType,
  ShadingType, PageNumber, Header, Footer, PageBreak, TableOfContents,
} = require("docx");

const CW = 9360; // content width (US Letter, 1" margins)
const HX = { style: BorderStyle.SINGLE, size: 1, color: "BBBBBB" };
const BORDERS = { top: HX, bottom: HX, left: HX, right: HX };
const HEAD_FILL = "D9E2F3";
const MARG = { top: 60, bottom: 60, left: 110, right: 110 };

function P(text, opts = {}) {
  return new Paragraph({
    spacing: { after: opts.after ?? 120, line: 276 },
    alignment: opts.align,
    children: [new TextRun({ text, bold: opts.bold, italics: opts.italics, size: opts.size })],
  });
}
function rich(runs, opts = {}) {
  return new Paragraph({ spacing: { after: opts.after ?? 120, line: 276 }, children: runs });
}
function H1(text) { return new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(text)] }); }
function H2(text) { return new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(text)] }); }
function bullet(text) {
  return new Paragraph({ numbering: { reference: "bul", level: 0 }, spacing: { after: 60, line: 276 },
    children: [new TextRun(text)] });
}
function cell(text, { w, header = false, bold = false, align } = {}) {
  return new TableCell({
    borders: BORDERS, width: { size: w, type: WidthType.DXA }, margins: MARG,
    shading: header ? { fill: HEAD_FILL, type: ShadingType.CLEAR } : undefined,
    children: [new Paragraph({ alignment: align,
      children: [new TextRun({ text, bold: header || bold, size: 20 })] })],
  });
}
function table(widths, rows) {
  return new Table({
    width: { size: CW, type: WidthType.DXA }, columnWidths: widths,
    rows: rows.map((r, ri) => new TableRow({
      children: r.map((c, ci) => cell(String(c), {
        w: widths[ci], header: ri === 0, align: ci === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
      })),
    })),
  });
}

const children = [];

// ---------- Title ----------
children.push(
  new Paragraph({ spacing: { before: 1200, after: 120 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Automated Risk Profiling of Public Companies", bold: true, size: 40 })] }),
  new Paragraph({ spacing: { after: 400 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Using Multi-Document RAG", bold: true, size: 40 })] }),
  new Paragraph({ spacing: { after: 120 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "CS 455 / CS 555 — Large Language Models — Spring 2025/2026", size: 24 })] }),
  new Paragraph({ spacing: { after: 120 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Project Category: (A) Application / System Building + (C) Evaluation", italics: true, size: 22 })] }),
  new Paragraph({ spacing: { before: 600, after: 80 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Authors: Eray Akalın  /  [Group Member 2]  /  [Group Member 3 — optional]", size: 22 })] }),
  new Paragraph({ spacing: { after: 80 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Final Project Report", size: 22 })] }),
  new Paragraph({ children: [new PageBreak()] }),
);

// ---------- TOC ----------
children.push(H1("Table of Contents"));
children.push(new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-2" }));
children.push(new Paragraph({ children: [new PageBreak()] }));

// ---------- Abstract ----------
children.push(H1("Abstract"));
children.push(P("Public companies disclose detailed risk information in the Item 1A “Risk Factors” section of their annual Form 10-K filings, but these documents are long, repetitive, and hard to compare across companies. We present an LLM-based financial document-intelligence system that automatically extracts, structures, and compares company-level risk profiles from SEC 10-K filings using a multi-document Retrieval-Augmented Generation (RAG) pipeline. For each of eight predefined risk categories, the system issues category-specific queries, retrieves and reranks evidence chunks, and prompts an instruction-tuned LLM under a strict JSON schema to produce a presence flag, a five-level severity estimate, a short explanation, supporting evidence snippets, and a retrieval-grounded confidence score. The outputs are aggregated into per-company risk profiles and surfaced through a Streamlit dashboard with heatmaps, top-risk views, an evidence explorer, and pairwise comparison. We evaluate the system on five large-cap companies along three axes: retrieval quality (Recall@5, MRR, nDCG@5 against a baseline ladder), risk-category detection (accuracy and macro-F1), and faithfulness (a verbatim grounding proxy, RAGAS LLM-judged metrics, and a 0–2 rubric). Reranking proves the single largest contributor to retrieval quality (MRR 0.58 → 0.85), category presence detection reaches 0.95 accuracy, and 97% of evidence snippets are verbatim-grounded in the source."));

// ---------- 1. Introduction ----------
children.push(H1("1. Introduction"));
children.push(H2("1.1 Problem Definition"));
children.push(P("Form 10-K annual reports contain standardized but lengthy risk disclosures, especially in Item 1A. Analysts and investors must read these manually and cannot easily compare risk exposure across firms. This project automatically extracts and compares company-level risk profiles from multiple 10-K reports. The system takes public filings as input, retrieves evidence from risk-related text, maps it onto a predefined financial risk taxonomy, and produces structured outputs containing risk categories, severity labels, explanations, and source snippets — rather than behaving like a generic document chatbot."));
children.push(H2("1.2 Motivation"));
children.push(P("Financial professionals routinely compare companies across dimensions such as supply-chain exposure, regulatory pressure, competition, cybersecurity, demand uncertainty, and macroeconomic sensitivity. Generic chatbots answer ad-hoc questions but rarely produce reusable structured outputs, explicit evidence, or measurable risk profiles. A risk-intelligence system that keeps every claim grounded in official filings can support equity research, portfolio analysis, and due diligence while reducing manual reading time. The problem is also interesting from an LLM standpoint because it combines retrieval-augmented generation, structured prompting, metadata-aware retrieval, evidence attribution, and systematic evaluation."));
children.push(H2("1.3 Contributions"));
children.push(bullet("An end-to-end multi-document RAG pipeline over SEC 10-K filings producing schema-constrained, evidence-backed risk profiles."));
children.push(bullet("A category-aware retrieval design combining multi-query expansion, cross-encoder reranking, and relevance gating, with a retrieval-derived confidence signal that replaces unreliable LLM self-reported confidence."));
children.push(bullet("A systematic evaluation: a retrieval baseline ladder (keyword vs. embedding vs. embedding+reranker vs. hybrid), category-detection accuracy and macro-F1, and a three-layer faithfulness analysis (verbatim grounding, RAGAS, and a 0–2 rubric)."));
children.push(bullet("An interactive Streamlit dashboard with on-demand live analysis (type any ticker, pick a fiscal year, and the full pipeline runs in real time to add that company) and multi-year comparison, in addition to risk heatmaps, top-risk views, an evidence explorer, and pairwise company comparison."));

// ---------- 2. System Architecture ----------
children.push(H1("2. System Architecture"));
children.push(P("The pipeline is a sequence of seven stages, each producing a persisted artifact so the system is reproducible and individually testable:"));
children.push(table([2900, 6460], [
  ["Stage", "Output"],
  ["10-K collection", "Official annual reports downloaded from SEC EDGAR"],
  ["Section extraction", "Item 1A Risk Factors text per company"],
  ["Chunking + metadata", "Token-aware chunks with company, year, section, chunk ID"],
  ["Embedding + FAISS", "Dense vector index for semantic retrieval"],
  ["Taxonomy retrieval", "Top-k evidence chunks per risk category and company"],
  ["LLM extraction", "JSON risk profile: presence, severity, explanation, evidence, confidence"],
  ["Dashboard + comparison", "Heatmap, top risks, evidence explorer, pairwise comparison"],
]));
children.push(new Paragraph({ spacing: { after: 120 } }));
children.push(P("Component choices are summarized below. The generation LLM is served via the Groq API for speed; a local Qwen2.5-3B-Instruct fallback is supported for fully offline operation.", {}));
children.push(table([3400, 5960], [
  ["Component", "Choice"],
  ["Embedding model", "BAAI/bge-small-en-v1.5 (384-dim, normalized)"],
  ["Vector index", "FAISS IndexFlatIP (cosine similarity)"],
  ["Reranker", "cross-encoder/ms-marco-MiniLM-L-6-v2"],
  ["Generation LLM", "Llama-3.1-8B-Instant (Groq API); Qwen2.5-3B-Instruct local fallback"],
  ["Chunking", "400 tokens, 80-token overlap (tiktoken cl100k_base), heading-aware"],
  ["Output", "Strict JSON schema, validated and repaired"],
]));

children.push(H2("2.1 Interactive Dashboard and Live On-Demand Analysis"));
children.push(P("The system is delivered as a Streamlit dashboard with four analytical views — a risk heatmap (company × category severity), top risks per company, an evidence explorer that links each assessment back to its source snippets, and pairwise company comparison. Beyond these precomputed views, the dashboard exposes two interactive capabilities:"));
children.push(bullet("Live on-demand analysis: the user types any ticker (e.g., NFLX, GOOGL, PLTR) and selects a fiscal year; the system then runs the entire pipeline in real time — downloading the 10-K from SEC EDGAR, extracting Item 1A, chunking, embedding, indexing, retrieving per category, and prompting the LLM — and adds the resulting risk profile to the dashboard. This generalizes the system beyond a fixed corpus to arbitrary public companies."));
children.push(bullet("Multi-year analysis: fiscal years 2021–2025 are selectable, and the comparator supports year-over-year severity comparison and per-category risk-trend tracking for a given company, enabling longitudinal risk analysis."));
children.push(P("This replaces the optional follow-up-question interface mentioned in the proposal with a more useful capability: on-demand, structured profiling of any company rather than free-form chat.", { italics: true }));

// ---------- 3. Dataset ----------
children.push(H1("3. Dataset"));
children.push(P("We use publicly available FY2025 Form 10-K filings obtained from SEC EDGAR. The evaluated dataset consists of five large-cap U.S. companies; the live-mode interface additionally supports arbitrary tickers on demand (e.g., GOOGL, NFLX, PLTR were profiled as demonstrations but are excluded from the evaluation set). We focus on Item 1A Risk Factors, because it contains standardized risk disclosures that are comparable across companies. All documents are public regulatory filings, so legal and ethical risk is low."));
children.push(table([2600, 3600, 1580, 1580], [
  ["Ticker", "Company", "Filing date", "Chunks"],
  ["AAPL", "Apple Inc.", "2025-10-31", "31"],
  ["TSLA", "Tesla, Inc.", "2026-01-29", "44"],
  ["NVDA", "NVIDIA Corporation", "2026-02-25", "63"],
  ["AMZN", "Amazon.com, Inc.", "2026-02-06", "33"],
  ["AMD", "Advanced Micro Devices", "2026-02-04", "62"],
  ["Total", "5 companies", "FY2025", "233"],
]));
children.push(new Paragraph({ spacing: { after: 60 } }));
children.push(rich([
  new TextRun({ text: "Note on Microsoft: ", bold: true }),
  new TextRun("The proposal listed Microsoft among example companies, but MSFT’s 10-K body contains no matchable “Item 1A / Risk Factors” heading (only a table-of-contents entry), so heading-based extraction fails. MSFT is therefore excluded; the dataset still satisfies the proposed 5–6 company scope. Restoring MSFT via structure-based extraction is noted as future work."),
]));

// ---------- 4. Risk Taxonomy ----------
children.push(H1("4. Risk Taxonomy"));
children.push(P("We define eight risk categories. Each category carries a description and three natural-language query templates used for retrieval (see Section 5.3)."));
[
  ["Supply Chain Risk", "Single-source suppliers, component shortages, manufacturing concentration, logistics."],
  ["Regulatory / Legal Risk", "Government regulation, litigation, investigations, fines, compliance failures."],
  ["Competition Risk", "Pricing pressure, market-share loss, new entrants, competitive disadvantage."],
  ["Cybersecurity Risk", "Data breaches, cyberattacks, ransomware, privacy violations."],
  ["Demand / Market Risk", "Demand uncertainty, shifting preferences, market saturation, revenue concentration."],
  ["Macroeconomic Risk", "Recession, inflation, interest rates, currency, geopolitical instability."],
  ["Operational Risk", "System outages, process failures, workforce, quality, business continuity."],
  ["IP / Technology Risk", "Patent disputes, technology obsolescence, R&D failure, AI governance."],
].forEach(([n, d], i) => children.push(rich([
  new TextRun({ text: `${i + 1}. ${n}: `, bold: true }), new TextRun(d)], { after: 60 })));

// ---------- 5. Methodology ----------
children.push(H1("5. Methodology"));
children.push(H2("5.1 Section Extraction and Chunking"));
children.push(P("Item 1A is located by collecting every “Item 1A … Risk Factors” candidate, keeping only standalone headings on their own line (discarding inline cross-references that a naive last-match heuristic would wrongly select), and selecting the candidate whose span to the next section is longest. Extracted text is split into token-aware chunks (400 tokens, 80-token overlap, measured with tiktoken) that preserve paragraph boundaries and start a fresh chunk at risk-factor sub-headings, keeping individual risk factors intact. Each chunk stores company, fiscal year, section, and a unique chunk ID."));
children.push(H2("5.2 Embedding and Indexing"));
children.push(P("Chunks are embedded with BAAI/bge-small-en-v1.5 (L2-normalized) and indexed in FAISS using inner product, which equals cosine similarity for normalized vectors. Models are loaded once per process and shared across the pipeline for efficiency."));
children.push(H2("5.3 Category-Aware Retrieval"));
children.push(P("For each risk category, three natural-language query templates are issued (e.g., “What cybersecurity, data breach, or cyberattack risks does the company face?”). Natural questions markedly outperform keyword-bag queries because the cross-encoder reranker is trained on natural query–document pairs. Results across templates are deduplicated and fused, then reranked with the cross-encoder. A relevance-gating step keeps only chunks whose reranked relevance is within a fixed ratio of the category’s best chunk (with an absolute noise floor), so the number of evidence chunks varies with the strength of the evidence instead of returning a fixed top-k. A hybrid BM25+dense variant (fused via Reciprocal Rank Fusion) was implemented but disabled by default after evaluation (Section 7.1)."));
children.push(H2("5.4 LLM Structured Extraction"));
children.push(P("The LLM never receives the full report — only the retrieved evidence chunks and a strict JSON schema. For each company and category it outputs: presence (boolean), a five-level severity (negligible / low / medium / high / critical), a 1–3 sentence explanation, direct-quote evidence snippets, and a confidence value. The prompt forbids relying on the boilerplate phrase “material adverse effect” as evidence of high severity and requires a concrete escalator (a quantified figure, a named active proceeding, or a realized incident) to justify high/critical; few-shot examples anchor the medium-vs-high and low-vs-medium boundaries. Responses are parsed and validated with JSON repair."));
children.push(H2("5.5 Confidence and Severity Calibration"));
children.push(P("Small instruction-tuned models collapse their self-reported confidence to a near-constant value. We therefore override LLM confidence with a retrieval-derived score combining the strongest chunk’s relevance, the mean relevance, and cross-query agreement; the model’s original value is retained for transparency. A weak-evidence guardrail caps severity at “low” when even the best retrieved chunk is only weakly relevant, and categories with no sufficiently relevant evidence short-circuit to “not present” without an LLM call (saving tokens and avoiding fabricated severity)."));

// ---------- 6. Evaluation Setup ----------
children.push(H1("6. Evaluation Setup"));
children.push(P("Evaluation covers retrieval quality, risk-category detection, and faithfulness, following the proposal’s plan."));
children.push(bullet("Retrieval: 40 company–risk queries (8 categories × 5 companies) with relevant chunk IDs annotated; metrics are Recall@5, MRR, and nDCG@5."));
children.push(bullet("Category detection: binary risk-presence over 40 company×category pairs; gold labels are source-verified (every category is genuinely discussed in these large-cap filings); metrics are accuracy and macro-F1."));
children.push(bullet("Faithfulness: (i) a verbatim grounding proxy that checks whether evidence snippets appear in the source text, (ii) RAGAS faithfulness and context-precision judged by Llama-3.3-70B-Versatile (a deliberately stronger model than the 8B generator, avoiding circular self-evaluation), and (iii) a 0–2 rubric mapped from RAGAS faithfulness."));
children.push(rich([new TextRun({ text: "Annotation note: ", bold: true }),
  new TextRun("Gold labels for retrieval relevance and category presence were produced as a model-assisted (“silver”) set and spot-checked. This is disclosed for transparency and listed as a limitation.")]));

// ---------- 7. Results ----------
children.push(H1("7. Results"));
children.push(H2("7.1 Retrieval Quality (baseline ladder)"));
children.push(P("We compare four configurations on the same 40 queries (top-5). Keyword search and dense FAISS are the two required baselines; adding the cross-encoder reranker is the key ablation, and a BM25+dense hybrid is an additional variant."));
children.push(table([3960, 1800, 1800, 1800], [
  ["Method", "MRR", "Recall@5", "nDCG@5"],
  ["Keyword (baseline)", "0.396", "0.468", "0.357"],
  ["Dense FAISS (no rerank)", "0.575", "0.406", "0.419"],
  ["Dense + Reranker", "0.850", "0.849", "0.811"],
  ["Hybrid (BM25+dense) + Rerank", "0.815", "0.808", "0.760"],
]));
children.push(new Paragraph({ spacing: { after: 80 } }));
children.push(P("The cross-encoder reranker is by far the largest contributor, lifting MRR from 0.58 to 0.85 and Recall@5 from 0.41 to 0.85 over dense retrieval alone. Keyword search is the weakest, as expected. The BM25+dense hybrid slightly underperforms pure dense+rerank: with natural-language queries and a strong dense embedding, lexical fusion mainly injects noise, so the hybrid path is disabled by default."));
children.push(H2("7.2 Risk-Category Detection"));
children.push(P("Binary risk-presence is evaluated over 39 scored company×category pairs (one excluded due to an LLM rate-limit failure)."));
children.push(table([4680, 4680], [
  ["Metric", "Value"],
  ["Accuracy", "0.949"],
  ["Macro-F1", "0.487"],
  ["Present class (P / R / F1, n=39)", "1.00 / 0.949 / 0.974"],
  ["Absent class (support)", "0 (no true negatives)"],
]));
children.push(new Paragraph({ spacing: { after: 80 } }));
children.push(P("Accuracy is high (0.949). Macro-F1 (0.487) is depressed because the dataset has no true negatives — every category is genuinely discussed in these comprehensive filings — while the model incorrectly marked two present risks as absent (AMZN Supply Chain and AMZN Cybersecurity). Macro-F1 thus correctly penalizes the model’s unreliable “absent” predictions, which stem from over-aggressive relevance gating (Section 8)."));
children.push(H2("7.3 Faithfulness and Generation Quality"));
children.push(P("Across the five evaluated companies, 75 of 77 evidence snippets (97.4%) appear verbatim in the source text, indicating very low fabricated-quote rates. RAGAS (judge: Llama-3.3-70B) reports faithfulness 0.61 and context-precision 0.89 over 15 scored samples; high context-precision confirms that retrieved chunks are relevant, while the lower faithfulness shows the generator sometimes adds claims to its explanation that exceed the evidence."));
children.push(table([4680, 2340, 2340], [
  ["RAGAS metric", "Score", "Scored"],
  ["Faithfulness", "0.607", "15/16"],
  ["Context precision (no ref.)", "0.894", "15/16"],
]));
children.push(new Paragraph({ spacing: { after: 80 } }));
children.push(P("Mapping RAGAS faithfulness to the proposed 0–2 rubric (≥0.8 → 2 fully supported, 0.3–0.8 → 1 partial, <0.3 → 0 unsupported) yields a mean of 1.13:"));
children.push(table([3120, 4120, 2120], [
  ["Rubric score", "Meaning", "Count"],
  ["2", "Fully supported", "4"],
  ["1", "Partially supported", "9"],
  ["0", "Unsupported / hallucinated", "2"],
  ["Mean", "—", "1.13"],
]));
children.push(new Paragraph({ spacing: { after: 80 } }));
children.push(P("Per-category RAGAS results highlight where generation is weakest. Context precision is uniformly high (retrieval is good), but faithfulness is lowest for Demand/Market and Macroeconomic risks and highest for Regulatory and Competition risks."));
children.push(table([4360, 1500, 1750, 1750], [
  ["Category", "n", "Faithfulness", "Context prec."],
  ["Demand / Market Risk", "2", "0.42", "1.00"],
  ["Macroeconomic Risk", "2", "0.50", "0.92"],
  ["Operational Risk", "2", "0.54", "0.88"],
  ["Cybersecurity Risk", "2", "0.55", "1.00"],
  ["Supply Chain Risk", "2", "0.62", "0.88"],
  ["IP / Technology Risk", "1", "0.67", "0.45"],
  ["Competition Risk", "2", "0.78", "1.00"],
  ["Regulatory / Legal Risk", "2", "0.81", "0.80"],
]));
children.push(H2("7.4 Severity Distribution"));
children.push(P("Across the 40 assessments of the five evaluated companies, severity is well spread rather than collapsing to a single label — evidence that the rubric and few-shot calibration counter the model’s default tendency to mark everything “high.”"));
children.push(table([4680, 4680], [
  ["Severity", "Count"],
  ["negligible", "3"],
  ["low", "2"],
  ["medium", "23"],
  ["high", "12"],
]));

// ---------- 8. Error Analysis ----------
children.push(H1("8. Error Analysis"));
children.push(bullet("False negatives from gating: AMZN Cybersecurity and AMZN Supply Chain are discussed in the filing (verified by keyword presence) but were marked “not present.” The relevance gate dropped weakly-ranked but genuinely relevant chunks; a lower floor or a category-specific threshold would recover them."));
children.push(bullet("Faithfulness gaps: low-faithfulness explanations (e.g., AAPL Supply Chain at 0.25) add plausible but unsupported claims even when the cited snippets are verbatim. Tightening the prompt to restrict explanations strictly to the quoted evidence, or post-hoc filtering low-faithfulness outputs, would help."));
children.push(bullet("Hybrid retrieval regression: BM25 fusion lowered ranking quality under natural-language queries, motivating the dense-only default."));
children.push(bullet("Rate-limit failure: one category (NVDA Operational) failed to score under the free-tier token limit; such failures are explicitly flagged rather than silently producing a default profile."));

// ---------- 9. Limitations ----------
children.push(H1("9. Limitations"));
children.push(bullet("Five companies and English-only Item 1A; Item 7/7A are not yet included."));
children.push(bullet("Microsoft is excluded due to heading-less section structure (extraction limitation)."));
children.push(bullet("Gold labels for retrieval relevance and category presence are model-assisted (“silver”) and spot-checked, not fully independent human annotations; metrics should be read accordingly."));
children.push(bullet("The category-detection set has no true negatives, so macro-F1 is dominated by class imbalance; accuracy and the false-negative analysis are the operative signals."));
children.push(bullet("Severity remains partly subjective; we evaluate category detection separately from severity scoring, as planned."));
children.push(bullet("The generation model is a small 8B model; faithfulness (mean rubric 1.13) leaves clear room for improvement."));

// ---------- 10. Future Work ----------
children.push(H1("10. Future Work"));
children.push(bullet("Structure-based extraction to restore Microsoft and other heading-less filings; extend to Item 7/7A."));
children.push(bullet("Independent human annotation to upgrade the silver gold sets to gold."));
children.push(bullet("Faithfulness-oriented decoding: stricter evidence-only explanations and post-hoc faithfulness filtering."));
children.push(bullet("Category-adaptive relevance gating to reduce false negatives."));
children.push(bullet("Year-over-year risk trend analysis and a larger company universe."));

// ---------- 11. Conclusion ----------
children.push(H1("11. Conclusion"));
children.push(P("We built and evaluated a multi-document RAG system that turns SEC 10-K Item 1A disclosures into structured, evidence-backed, comparable risk profiles. The evaluation confirms the design choices: cross-encoder reranking is the dominant driver of retrieval quality (MRR 0.58 → 0.85), category presence is detected at 0.95 accuracy, and 97% of evidence snippets are verbatim-grounded, with RAGAS context-precision of 0.89. The analysis also exposes concrete, actionable weaknesses — gating-induced false negatives and explanation faithfulness — that define a clear path forward. The system meets the proposal’s goal of producing reusable, measurable, evidence-grounded risk intelligence rather than a generic document chatbot."));

// ---------- References ----------
children.push(H1("References"));
[
  "[1] SEC Investor.gov. How to Read a 10-K. https://www.investor.gov/introduction-investing/getting-started/researching-investments/how-read-10-k",
  "[2] SEC. EDGAR Application Programming Interfaces. https://www.sec.gov/search-filings/edgar-application-programming-interfaces",
  "[3] FAISS Documentation. Similarity search and clustering of dense vectors. https://faiss.ai",
  "[4] Qwen Team. Qwen2.5-3B-Instruct model card. https://huggingface.co/Qwen/Qwen2.5-3B-Instruct",
  "[5] BAAI. bge-small-en-v1.5 model card. https://huggingface.co/BAAI/bge-small-en-v1.5",
  "[6] Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. https://arxiv.org/abs/2005.11401",
  "[7] Es et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation. https://arxiv.org/abs/2309.15217",
].forEach(r => children.push(P(r, { after: 60, size: 20 })));

// ---------- Document ----------
const doc = new Document({
  creator: "CS455 Project",
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, font: "Arial", color: "1F3864" },
        paragraph: { spacing: { before: 280, after: 160 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 25, bold: true, font: "Arial", color: "2E5496" },
        paragraph: { spacing: { before: 200, after: 120 }, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [
    { reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•",
      alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 260 } } } }] },
  ] },
  sections: [{
    properties: { page: {
      size: { width: 12240, height: 15840 },
      margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
    } },
    footers: { default: new Footer({ children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: "Automated Risk Profiling with Multi-Document RAG  —  ", size: 18 }),
                 new TextRun({ children: ["Page ", PageNumber.CURRENT], size: 18 })],
    })] }) },
    children,
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync("CS455_Final_Report_Risk_Profiling_RAG.docx", buf);
  console.log("Report written: CS455_Final_Report_Risk_Profiling_RAG.docx");
});
