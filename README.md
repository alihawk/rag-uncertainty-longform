# Uncertainty Estimation for Long-Form RAG

This repository implements and evaluates uncertainty estimation methods for Retrieval-Augmented Generation (RAG) systems in long-form text generation scenarios.

## Overview

RAG systems combine document retrieval with large language models to improve factual accuracy, but they can still produce incorrect or unsupported outputs. This project investigates whether uncertainty estimation methods can effectively identify when RAG outputs should not be trusted.

We evaluate three uncertainty estimation approaches:
- **MARS** (white-box): Meaning-Aware Response Scoring
- **Eccentricity** (black-box): Distance-based uncertainty via embedding space
- **SEEW** (black-box, novel): Semantic Entropy with Evidence Weighting

## Repository Structure

```
├── bibliography/          # Citation files
├── journal/              # Draft manuscripts and write-ups
├── presentation/         # Slide decks and presentation assets
├── report/              # Final PDF report and figures
└── src/
    ├── configs/         # Configuration files
    ├── data/           # Data storage
    ├── generator/      # LLM generation code (Qwen2.5-7B)
    ├── index/          # BM25 index files
    ├── retriever/      # BM25 + cross-encoder reranking
    ├── reports/        # Benchmark results
    ├── scripts/        # Utility scripts (SAFE runner, etc.)
    └── ue/            # Uncertainty estimation implementations
```

## Quick Setup

Clone the repository:
```bash
git clone https://github.com/yourusername/rag-uncertainty-longform.git
cd rag-uncertainty-longform
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Pipeline Components

### Retrieval System
- **Corpus**: Wikipedia-based corpus (wiki-18-corpus)
- **Method**: BM25 indexing via Pyserini + MS-Marco-MiniLM-L6-v2 cross-encoder reranking
- **Output**: Top-3 reranked passages as context

### Generator
- **Model**: Qwen2.5-7B-Instruct (full precision and 4-bit quantized versions)
- **Framework**: HuggingFace Transformers

### Evaluation
- **Dataset**: FactScore-BIO (50 randomly sampled biographies)
- **Factuality Metric**: SAFE (claim-level verification)
- **UE Metrics**: AUROC and PRR (Prediction-Rejection Ratio)

## Key Findings

The results indicate that uncertainty estimation methods show weak correlation with factual correctness in long-form RAG:

| Method | AUROC (Full) | PRR (Full) | AUROC (4bit) | PRR (4bit) |
|--------|--------------|------------|--------------|------------|
| **MARS** | 0.601 | 0.254 | 0.555 | 0.206 |
| **Eccentricity** | 0.532 | 0.052 | 0.501 | -0.022 |
| **SEEW** | 0.556 | 0.084 | 0.526 | 0.079 |

All methods perform only marginally better than random chance, suggesting that uncertainty estimation for long-form, claim-heavy generation remains challenging.

## Implementation Notes

- Modified [TruthTorchLM](https://github.com/Rijkkie/TruthTorchLM) library to use custom retrieval instead of SerperAPI
- Added context support to Question-Answer Generation for claim-level uncertainty
- Implemented SEEW as a lightweight black-box method with semantic clustering and evidence weighting

## References

Based on research from:
- MARS: Bakman et al. (ACL 2024)
- Eccentricity: Lin et al. (TMLR 2024)
- SAFE: Wei et al. (NeurIPS 2024)
- TruthTorchLM: Yaldiz et al. (EMNLP 2025)

## Authors

- Muhammad Ali (Radboud University)
- Rijk Kregting (Radboud University)
- Adith Shivakumar (Radboud University)

## License

This project is part of the Information Retrieval 2025 course at Radboud University.
