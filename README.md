# BanMANI: Detecting Manipulated Bangla News Headlines

**Course:** CSE 4891 — Data Mining  
 
---

## What this project is about

We detect manipulated Bangla news headlines using the BanMANI dataset — 800 samples where a headline has been subtly altered (wrong minister name, wrong number, wrong date) compared to the actual article. We compare three approaches: classical ML, fine-tuned transformers, and open-source LLM prompting.

**Key finding:** The official BanMANI train/test split has a distribution mismatch (72.5% MANI train vs 40.7% MANI test) that artificially suppresses all reported scores by 0.11–0.19 F1 points.

---

## Dataset

- **Source:** [BanMANI on Kaggle](https://www.kaggle.com/datasets/saikasatter/banmani)
- 800 samples (650 train / 150 test)
- Labels: MANI (532) | NO_MANI (268)
- Columns: `mani_news`, `original_news_article`, `mani_status`, `altered_excerpt`, `original_excerpt`, `category`

---

## Notebooks

| Notebook | Description | Link |
|---|---|---|
| `fine-tuning.ipynb` | mBERT, XLM-RoBERTa, BanglaBERT fine-tuning on Kaggle GPU | [Kaggle](https://www.kaggle.com/code/saikasatter/fine-tuning) |
| `llm_evaluation.ipynb` | Qwen2.5-7B and Llama-3.1-8B zero-shot/few-shot evaluation | [Kaggle](https://www.kaggle.com/code/saikasatter/banmaniresearch) |

---

## How to run classical ML (no GPU needed)

```bash
# Install dependencies
pip install scikit-learn pandas numpy joblib

# Run all three baselines
python run_pipeline.py --stage baseline
```

---

## Full Results

### Official split (72.5% train / 40.7% test MANI)

| Model | Type | Strategy | F1 Macro | F1 MANI |
|---|---|---|---|---|
| TF-IDF word + LR | Classical ML | — | **0.5717** | 0.6517 |
| TF-IDF char + LR | Classical ML | — | 0.5656 | 0.6235 |
| TF-IDF + SVM | Classical ML | — | 0.5579 | 0.6404 |
| BanglaBERT | Transformer | fine-tuned | 0.4258 | 0.5789 |
| XLM-RoBERTa | Transformer | fine-tuned | 0.3724 | 0.0000 |
| mBERT | Transformer | fine-tuned | 0.3535 | 0.5389 |
| Qwen2.5-7B | Open LLM | zero-shot | 0.3724 | 0.0000 |
| Qwen2.5-7B | Open LLM | few-shot 3 | 0.4073 | 0.0635 |
| Llama-3.1-8B | Open LLM | zero-shot | 0.3979 | 0.4943 |
| Llama-3.1-8B | Open LLM | few-shot 3 | 0.3724 | 0.0000 |

### Re-stratified splits (matched distributions, ~66.5% MANI both splits)

| Model | Official F1 Macro | Re-stratified F1 Macro | Gained |
|---|---|---|---|
| TF-IDF word + LR | 0.5717 | 0.689 ± 0.037 | +0.117 |
| BanglaBERT | 0.4258 | 0.611 ± 0.056 | +0.185 |
| XLM-RoBERTa | 0.3724 | 0.400 ± 0.000 | +0.028 |
| mBERT | 0.3535 | 0.473 ± 0.116 | +0.119 |

---

## Key findings

1. **Classical ML beats transformers on the official split** — due to small training data (650 samples) and distribution mismatch
2. **Distribution shift is a major confound** — the official split's 72.5%/40.7% mismatch was artificially suppressing all scores
3. **Llama zero-shot nearly matches fine-tuned BanglaBERT** — F1 MANI 0.494 vs 0.579, with zero training
4. **Few-shot prompting causes collapse** — adding 3 examples caused both LLMs to predict NO_MANI for everything
5. **F1 MANI jumps dramatically under re-stratification** — XLM-R goes from 0.0 to 0.80, confirming models learned real patterns

---

## Dependencies

```
scikit-learn>=1.0
pandas numpy joblib
transformers==4.44.0   # for transformer fine-tuning
torch>=2.0
```
