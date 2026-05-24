# Evaluation Report — FY2025
_Generated: 2026-05-25_

## Retrieval Metrics
Labeled queries: 40 | Top-K: 5
(Ground truth: LLM-labeled silver set — spot-check recommended.)

| Method | MRR | Recall@K | nDCG@K |
|---|---|---|---|
| HYBRID (BM25+dense) | 0.7883 | 0.7839 | 0.7319 |
| DENSE-only | 0.8300 | 0.8297 | 0.7901 |

## Generation Quality
- **Grounding (faithfulness proxy):** 101/104 snippets verbatim in source (**97.1%**)
- **Mean confidence:** 0.646
- **Mean evidence chunks:** 1.89
- **Extraction failures:** 1

### Severity distribution
| Severity | Count |
|---|---|
| negligible | 6 |
| low | 3 |
| medium | 31 |
| high | 16 |

### Ungrounded snippets (possible paraphrase/hallucination)
- [AMD] Regulatory / Legal Risk: "We may incur costs and resources in order to comply with various new or proposed climate-r…"
- [AMD] Competition Risk: "Intel uses its microprocessor market position to price its products aggressively and targe…"
- [GOOGL] Supply Chain Risk: "In addition, manufacturing and supply of servers and network equipment for our technical i…"
