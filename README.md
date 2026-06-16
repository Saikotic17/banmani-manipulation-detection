# BanMANI Manipulation Detection — Research Pipeline

## Dataset
- **BanMANI.csv** — 800 Bangla news samples (650 train / 150 test)
- Labels: `MANI` (532) | `NO_MANI` (268)
- Columns: `category`, `data_type`, `mani_status`, `altered_excerpt`, `original_excerpt`, `mani_news`, `original_news_article`

## Project Structure
```
banmani_research/
├── data/
│   ├── BanMANI.csv
│   └── preprocess.py          ← Bangla normalization + text combination
├── models/
│   ├── baseline_tfidf.py      ← TF-IDF + LR / SVM (no GPU needed)
│   ├── transformer_finetune.py← mBERT / XLM-R / BanglaBERT fine-tuning
│   └── llm_evaluation.py      ← Claude API zero-shot / few-shot / CoT
├── results/
│   ├── analysis.py            ← Error analysis + comparison tables
│   └── outputs/               ← Saved results (JSON, LaTeX)
└── run_pipeline.py            ← Main CLI entry point
```

## Quick Start

### 1. Baseline (no GPU, no API key required)
```bash
python run_pipeline.py --stage baseline
```

### 2. LLM Evaluation (requires ANTHROPIC_API_KEY)
```bash
export ANTHROPIC_API_KEY=sk-ant-...
python run_pipeline.py --stage llm --mode zero_shot
python run_pipeline.py --stage llm --mode few_shot_3
python run_pipeline.py --stage llm --mode few_shot_5
python run_pipeline.py --stage llm --mode cot
# Limit samples for cost control:
python run_pipeline.py --stage llm --mode few_shot_5 --max-samples 50
```

### 3. Transformer Fine-tuning (requires GPU + pip install transformers torch)
```bash
pip install transformers torch
python run_pipeline.py --stage transformer --model xlmr --epochs 5
python run_pipeline.py --stage transformer --model banglabert --epochs 5
```

### 4. Analysis & Comparison
```bash
python run_pipeline.py --stage analysis
```

## Models Supported
| Key | Model | Notes |
|-----|-------|-------|
| `mbert` | bert-base-multilingual-cased | 104-language BERT |
| `xlmr` | xlm-roberta-base | Best cross-lingual baseline |
| `banglabert` | sagorsarker/bangla-bert-base | Domain-specific, best expected |
| `xlmr_large` | xlm-roberta-large | Highest capacity |

## Key Design Decisions
- **Text combination**: `[CLAIM] <headline> [ARTICLE] <article>` for transformers; concatenation for TF-IDF
- **Class weighting**: balanced weights applied to handle 72.5% MANI skew in training
- **Bangla normalization**: zero-width char removal, nukta normalization, whitespace collapse
- **Few-shot selection**: balanced strategy (k/2 MANI + k/2 NO_MANI)

## Baseline Results (Computed)
| Model | Accuracy | F1 Macro | F1 MANI |
|-------|----------|----------|---------|
| TF-IDF word+LR | 0.587 | 0.572 | 0.652 |
| TF-IDF char+LR | 0.573 | 0.566 | 0.624 |
| TF-IDF+SVM | 0.573 | 0.558 | 0.640 |

## Dependencies
```
scikit-learn>=1.0
pandas, numpy
transformers>=4.30 (for transformers stage)
torch>=2.0 (for transformers stage)
anthropic>=0.20 (for LLM stage)
```
