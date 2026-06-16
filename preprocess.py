"""
BanMANI Data Preprocessing Module
Handles Bangla text cleaning, normalization, and feature construction.
"""

import re
import pandas as pd
import numpy as np
from typing import Optional, Tuple


# ── Bangla Unicode ranges ────────────────────────────────────────────────────
BANGLA_RANGE = r'[\u0980-\u09FF]'
BANGLA_PUNCT = r'[\u0964\u0965]'          # Dari / double dari
ZERO_WIDTH   = r'[\u200B-\u200D\uFEFF]'  # Zero-width chars common in web text


def normalize_bangla(text: str) -> str:
    """
    Normalize Bangla text:
    - Remove zero-width characters
    - Normalize nukta variants
    - Collapse whitespace
    - Strip leading/trailing spaces
    """
    if not isinstance(text, str):
        return ""
    # Remove zero-width noise
    text = re.sub(ZERO_WIDTH, '', text)
    # Normalize ড় (0x09DC → 0x09A1 + 0x09BC) edge cases
    text = text.replace('\u09A1\u09BC', '\u09DC')
    text = text.replace('\u09A2\u09BC', '\u09DD')
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def combine_claim_and_article(row: pd.Series, strategy: str = "sep_token") -> str:
    """
    Combine mani_news (claim/headline) with original_news_article (body).

    Strategies:
        'sep_token'   : [CLAIM] <claim> [ARTICLE] <article>  ← default for BERT/XLM-R
        'concat'      : claim + ' ' + article                ← for TF-IDF
        'claim_only'  : only mani_news                       ← ablation
        'excerpt'     : altered_excerpt if MANI else claim   ← fine-grained focus
    """
    claim   = normalize_bangla(row.get('mani_news', ''))
    article = normalize_bangla(row.get('original_news_article', ''))
    excerpt = normalize_bangla(row.get('altered_excerpt', ''))

    if strategy == "sep_token":
        return f"[CLAIM] {claim} [ARTICLE] {article}"
    elif strategy == "concat":
        return f"{claim} {article}"
    elif strategy == "claim_only":
        return claim
    elif strategy == "excerpt":
        return f"[CLAIM] {claim} [EXCERPT] {excerpt}" if excerpt else f"[CLAIM] {claim} [ARTICLE] {article}"
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def load_dataset(csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load BanMANI CSV and return (train_df, test_df).
    Adds:
        - 'label'   : int  (1=MANI, 0=NO_MANI)
        - 'text_*'  : combined text for each strategy
    """
    df = pd.read_csv(csv_path)

    # Binary label
    df['label'] = (df['mani_status'] == 'MANI').astype(int)

    # Build combined texts for all strategies
    for strategy in ['sep_token', 'concat', 'claim_only', 'excerpt']:
        col = f'text_{strategy}'
        df[col] = df.apply(lambda r: combine_claim_and_article(r, strategy), axis=1)

    # Char / token length features (useful for error analysis)
    df['claim_len']   = df['mani_news'].fillna('').apply(len)
    df['article_len'] = df['original_news_article'].fillna('').apply(len)
    df['has_excerpt'] = df['altered_excerpt'].notna().astype(int)

    train_df = df[df['data_type'] == 'TRAIN'].copy().reset_index(drop=True)
    test_df  = df[df['data_type'] == 'TEST'].copy().reset_index(drop=True)

    return train_df, test_df


def get_class_weights(labels: pd.Series) -> dict:
    """
    Compute inverse-frequency class weights to handle imbalance.
    Returns dict suitable for sklearn's class_weight param.
    """
    counts = labels.value_counts()
    total  = len(labels)
    return {cls: total / (len(counts) * cnt) for cls, cnt in counts.items()}


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    csv = sys.argv[1] if len(sys.argv) > 1 else "BanMANI.csv"
    train, test = load_dataset(csv)

    print(f"Train: {len(train)} rows  |  MANI={train['label'].sum()}  NO_MANI={(train['label']==0).sum()}")
    print(f"Test : {len(test)} rows   |  MANI={test['label'].sum()}  NO_MANI={(test['label']==0).sum()}")
    print("\nSample combined text (sep_token):")
    print(train['text_sep_token'].iloc[0][:300])
    print("\nClass weights:", get_class_weights(train['label']))
