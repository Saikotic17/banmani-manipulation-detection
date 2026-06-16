# banmani-manipulation-detection
Bangla News Manipulation Detection (BanMANI) — CSE 3812 Research Project
# BanMANI: Bangla News Manipulation Detection

**Course:** CSE 3812 — Artificial Intelligence  
**Dataset:** BanMANI (800 Bangla news samples, binary: MANI / NO_MANI)

## Project Overview
This project evaluates multiple approaches for detecting manipulated Bangla news claims:
- Stage 1: Classical ML (TF-IDF + Logistic Regression / LinearSVC)
- Stage 2: Transformer Fine-tuning (mBERT, XLM-RoBERTa, BanglaBERT)
- Stage 3: Open-Source LLM Zero/Few-Shot (Qwen2.5-7B, Llama-3.1-8B)

## Results Summary

| Model | Type | F1 Macro | F1 MANI |
|---|---|---|---|
| TF-IDF word + LR | Classical ML | 0.5717 | 0.6517 |
| TF-IDF char + LR | Classical ML | 0.5656 | 0.6235 |
| TF-IDF + LinearSVC | Classical ML | 0.5579 | 0.6404 |
| BanglaBERT | Transformer | 0.4258 | 0.5789 |
| Llama-3.1-8B zero-shot | LLM | 0.3979 | 0.4943 |
| XLM-RoBERTa | Transformer | 0.3724 | 0.0000 |
| mBERT | Transformer | 0.3535 | 0.5389 |

**Key finding:** Classical ML outperformed all transformer and LLM approaches on this 
small dataset (650 training samples), consistent with the small-data-regime hypothesis.

## How to Run

### Stage 1 — Classical ML Baseline (local)
```bash
pip install scikit-learn pandas numpy joblib
python run_pipeline.py --stage baseline
```

### Stage 2 — Transformer Fine-tuning (Kaggle recommended)
```bash
python run_pipeline.py --stage transformer --model banglabert
python run_pipeline.py --stage transformer --model xlmr
python run_pipeline.py --stage transformer --model mbert
```

### Stage 3 — LLM Evaluation (Kaggle, GPU required)
```bash
python run_pipeline.py --stage llm --mode zero_shot
python run_pipeline.py --stage llm --mode few_shot_3
```

## Dataset
BanMANI.csv is not included in this repository. 
Available at: [add your dataset source link here]

## Environment
- Python 3.10+
- transformers==4.44.0
- torch, scikit-learn, pandas, numpy
- Kaggle Tesla T4 GPU (Stages 2 & 3)
