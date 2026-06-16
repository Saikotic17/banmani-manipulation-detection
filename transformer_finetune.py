"""
Transformer Fine-tuning Module
Supports: bert-base-multilingual-cased, xlm-roberta-base, sagorsarker/bangla-bert-base
Fine-tunes on BanMANI for binary manipulation detection.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data.preprocess import load_dataset

# ── Lazy imports (only if torch/transformers available) ──────────────────────
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        EarlyStoppingCallback,
        DataCollatorWithPadding,
    )
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  transformers/torch not installed. Run: pip install transformers torch")


# ── Model configs ─────────────────────────────────────────────────────────────
MODEL_CONFIGS = {
    "mbert": {
        "name": "bert-base-multilingual-cased",
        "max_len": 512,
        "description": "Multilingual BERT — 104 languages, strong cross-lingual baseline",
    },
    "xlmr": {
        "name": "xlm-roberta-base",
        "max_len": 512,
        "description": "XLM-RoBERTa — stronger than mBERT on low-resource languages",
    },
    "banglabert": {
        "name": "sagorsarker/bangla-bert-base",
        "max_len": 512,
        "description": "BanglaBERT — Bangla-specific BERT, best in-domain performance expected",
    },
    "xlmr_large": {
        "name": "xlm-roberta-large",
        "max_len": 512,
        "description": "XLM-RoBERTa Large — most parameters, highest capacity",
    },
}


# ── Dataset Class ─────────────────────────────────────────────────────────────
class BanMANIDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int],
                 tokenizer, max_len: int = 512):
        self.texts    = texts
        self.labels   = labels
        self.tokenizer = tokenizer
        self.max_len  = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            truncation=True,
            padding=False,      # DataCollatorWithPadding handles this
            return_tensors=None,
        )
        enc['labels'] = self.labels[idx]
        return enc


# ── Metrics Function ──────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    preds = np.argmax(logits, axis=-1)

    acc   = accuracy_score(labels, preds)
    f1_m  = f1_score(labels, preds, average='macro')
    f1_w  = f1_score(labels, preds, average='weighted')
    f1_mani = f1_score(labels, preds, pos_label=1)
    try:
        auc = roc_auc_score(labels, probs[:, 1])
    except Exception:
        auc = 0.0

    return {
        'accuracy': acc,
        'f1_macro': f1_m,
        'f1_weighted': f1_w,
        'f1_mani': f1_mani,
        'roc_auc': auc,
    }


# ── Training Configuration ────────────────────────────────────────────────────
@dataclass
class FineTuneConfig:
    model_key: str = "xlmr"            # Key into MODEL_CONFIGS
    text_col: str = "text_sep_token"   # Which combined text to use
    num_epochs: int = 5
    batch_size: int = 8                # Reduce to 4 on CPU / small GPU
    grad_accum: int = 4                # Effective batch = batch_size * grad_accum
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    fp16: bool = False                 # Enable on CUDA
    save_dir: str = "results/transformer"
    eval_strategy: str = "epoch"
    load_best_model: bool = True
    early_stopping_patience: int = 2


def fine_tune_transformer(
    csv_path: str,
    config: FineTuneConfig = None,
) -> dict:
    """
    Fine-tune a transformer model on BanMANI.

    Returns dict with final evaluation metrics.
    """
    if not TRANSFORMERS_AVAILABLE:
        print("❌ Cannot fine-tune: transformers/torch not available.")
        return {}

    if config is None:
        config = FineTuneConfig()

    model_info = MODEL_CONFIGS[config.model_key]
    model_name = model_info["name"]
    os.makedirs(config.save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Fine-tuning: {model_name}")
    print(f"  {model_info['description']}")
    print(f"{'='*60}")

    # ── Load Data ─────────────────────────────────────────────────────────────
    print(f"\n📂 Loading data from {csv_path} ...")
    train_df, test_df = load_dataset(csv_path)

    X_train = train_df[config.text_col].tolist()
    y_train = train_df['label'].tolist()
    X_test  = test_df[config.text_col].tolist()
    y_test  = test_df['label'].tolist()

    print(f"   Train: {len(X_train)} samples | Test: {len(X_test)} samples")

    # ── Tokenizer & Datasets ──────────────────────────────────────────────────
    print(f"\n🔤 Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = BanMANIDataset(X_train, y_train, tokenizer, model_info['max_len'])
    test_dataset  = BanMANIDataset(X_test,  y_test,  tokenizer, model_info['max_len'])
    collator      = DataCollatorWithPadding(tokenizer=tokenizer)

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"\n🤖 Loading model: {model_name}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Device: {device}")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: 'NO_MANI', 1: 'MANI'},
        label2id={'NO_MANI': 0, 'MANI': 1},
    )

    # Class weights for imbalanced dataset
    class_counts = np.bincount(y_train)
    class_weights = torch.tensor(
        len(y_train) / (2 * class_counts), dtype=torch.float
    ).to(device)
    print(f"   Class weights: {class_weights.tolist()}")

    # ── Training Arguments ────────────────────────────────────────────────────
    output_dir = os.path.join(config.save_dir, config.model_key)
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size * 2,
        gradient_accumulation_steps=config.grad_accum,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        fp16=config.fp16 and torch.cuda.is_available(),
        evaluation_strategy=config.eval_strategy,
        save_strategy=config.eval_strategy,
        load_best_model_at_end=config.load_best_model,
        metric_for_best_model='f1_macro',
        greater_is_better=True,
        logging_steps=50,
        report_to='none',          # Disable wandb
        seed=42,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=config.early_stopping_patience
        )],
    )

    print(f"\n🚀 Starting training ...")
    trainer.train()

    # ── Evaluation ────────────────────────────────────────────────────────────
    print(f"\n📊 Evaluating on test set ...")
    eval_results = trainer.evaluate()
    print(json.dumps({k: round(v, 4) for k, v in eval_results.items()
                      if not k.startswith('eval_runtime')}, indent=2))

    # Save model & tokenizer
    model_save_path = os.path.join(output_dir, 'best_model')
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"\n✅ Model saved to: {model_save_path}")

    # Save results
    results = {
        'model': model_name,
        'config': config.__dict__,
        'eval': {k: round(v, 4) for k, v in eval_results.items()},
    }
    results_path = os.path.join(config.save_dir, f'{config.model_key}_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    return results


# ── Inference Helper ──────────────────────────────────────────────────────────
def predict_single(text: str, model_path: str) -> dict:
    """Run inference on a single text sample."""
    if not TRANSFORMERS_AVAILABLE:
        return {}
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model     = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0].tolist()
    pred  = int(torch.argmax(logits))

    return {
        'prediction': 'MANI' if pred == 1 else 'NO_MANI',
        'confidence': round(max(probs), 4),
        'prob_mani': round(probs[1], 4),
        'prob_no_mani': round(probs[0], 4),
    }


# ── CLI Entry ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    csv   = sys.argv[1] if len(sys.argv) > 1 else "../data/BanMANI.csv"
    mkey  = sys.argv[2] if len(sys.argv) > 2 else "xlmr"
    cfg   = FineTuneConfig(model_key=mkey, num_epochs=3)
    results = fine_tune_transformer(csv, cfg)
