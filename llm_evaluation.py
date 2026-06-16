"""
LLM Evaluation Module — Claude API (Prompt-based & Few-shot)
Evaluates Claude's ability to detect subtle manipulation in Bangla news.

Supports:
    - Zero-shot prompting
    - Few-shot prompting (k=3 and k=5)
    - Chain-of-thought (CoT)
    - Structured JSON output
"""

import os
import sys
import json
import time
import random
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data.preprocess import load_dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# ── Anthropic client ──────────────────────────────────────────────────────────
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("⚠️  anthropic package not installed. Run: pip install anthropic")


# ── Prompt Templates ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """আপনি একজন বাংলা সংবাদ বিশ্লেষক। আপনার কাজ হলো একটি সংবাদ শিরোনাম/দাবি এবং সংশ্লিষ্ট সংবাদ নিবন্ধ বিশ্লেষণ করে নির্ধারণ করা যে শিরোনামটি ম্যানিপুলেটেড (তথ্য বিকৃত) কিনা।

You are a Bangla news analyst. Your task is to analyze a news headline/claim and its associated article to determine if the headline has been manipulated (information distorted).

Manipulation includes: substituting names, locations, roles, numbers, or dates to change the meaning.

Always respond with a JSON object only: {"label": "MANI" or "NO_MANI", "confidence": 0.0-1.0, "reason": "brief explanation in English"}"""


ZERO_SHOT_TEMPLATE = """Claim/Headline:
{claim}

News Article:
{article}

Analyze if the claim has been manipulated compared to the article. Respond with JSON only."""


FEW_SHOT_TEMPLATE = """Here are {k} examples of manipulation detection:

{examples}

---
Now analyze this new instance:

Claim/Headline:
{claim}

News Article:
{article}

Respond with JSON only."""


COT_TEMPLATE = """Claim/Headline:
{claim}

News Article:
{article}

Think step by step:
1. What key facts are in the article? (persons, locations, numbers, roles)
2. What does the claim state?
3. Is there any discrepancy?
4. Final verdict?

Respond with JSON only: {{"label": "MANI" or "NO_MANI", "confidence": 0.0-1.0, "reason": "..."}}"""


# ── Few-shot Example Selection ────────────────────────────────────────────────
def select_few_shot_examples(
    train_df: pd.DataFrame,
    k: int = 3,
    strategy: str = "balanced",
    seed: int = 42,
) -> List[dict]:
    """
    Select k examples for few-shot prompting.

    Strategies:
        'balanced'  : k/2 MANI, k/2 NO_MANI
        'random'    : random k samples
        'diverse'   : one per category (varied manipulation types)
    """
    random.seed(seed)
    np.random.seed(seed)

    if strategy == "balanced":
        k_mani   = k // 2
        k_nomani = k - k_mani
        mani_rows   = train_df[train_df['label'] == 1].sample(k_mani, random_state=seed)
        nomani_rows = train_df[train_df['label'] == 0].sample(k_nomani, random_state=seed)
        examples_df = pd.concat([mani_rows, nomani_rows]).sample(frac=1, random_state=seed)
    elif strategy == "diverse":
        # One per category, balanced labels
        chosen = []
        for cat in train_df['category'].unique():
            sub = train_df[train_df['category'] == cat]
            for lbl in [1, 0]:
                sub_l = sub[sub['label'] == lbl]
                if len(sub_l) > 0:
                    chosen.append(sub_l.sample(1, random_state=seed).iloc[0])
                    if len(chosen) >= k:
                        break
            if len(chosen) >= k:
                break
        examples_df = pd.DataFrame(chosen[:k])
    else:  # random
        examples_df = train_df.sample(k, random_state=seed)

    examples = []
    for _, row in examples_df.iterrows():
        examples.append({
            'claim':   row['mani_news'],
            'article': row['original_news_article'][:800],  # Truncate for prompt length
            'label':   row['mani_status'],
            'excerpt': row.get('altered_excerpt', ''),
        })
    return examples


def format_examples(examples: List[dict]) -> str:
    """Format few-shot examples into prompt text."""
    lines = []
    for i, ex in enumerate(examples, 1):
        lines.append(f"Example {i}:")
        lines.append(f"  Claim: {ex['claim']}")
        lines.append(f"  Article (excerpt): {ex['article'][:400]}...")
        lines.append(f"  Label: {ex['label']}")
        if ex['label'] == 'MANI' and ex.get('excerpt'):
            lines.append(f"  Manipulation: altered text was '{ex['excerpt']}'")
        lines.append("")
    return "\n".join(lines)


# ── Claude API Caller ─────────────────────────────────────────────────────────
def call_claude(
    client,
    user_message: str,
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 300,
    retries: int = 3,
    retry_delay: float = 2.0,
) -> Optional[str]:
    """Call Claude API with retry logic."""
    for attempt in range(retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            return response.content[0].text.strip()
        except Exception as e:
            if attempt < retries - 1:
                print(f"   ⚠️  API error (attempt {attempt+1}): {e}. Retrying...")
                time.sleep(retry_delay * (attempt + 1))
            else:
                print(f"   ❌ API error after {retries} attempts: {e}")
                return None


def parse_response(response_text: Optional[str]) -> dict:
    """Parse JSON response from Claude."""
    if not response_text:
        return {'label': 'NO_MANI', 'confidence': 0.5, 'reason': 'API error'}
    try:
        # Strip markdown code blocks if present
        text = response_text.replace('```json', '').replace('```', '').strip()
        result = json.loads(text)
        label = result.get('label', 'NO_MANI').upper()
        if label not in ('MANI', 'NO_MANI'):
            label = 'NO_MANI'
        return {
            'label': label,
            'confidence': float(result.get('confidence', 0.5)),
            'reason': result.get('reason', ''),
        }
    except json.JSONDecodeError:
        # Fallback: look for MANI keyword
        label = 'MANI' if 'MANI' in response_text.upper() else 'NO_MANI'
        return {'label': label, 'confidence': 0.5, 'reason': response_text[:200]}


# ── Main Evaluation Loop ──────────────────────────────────────────────────────
def evaluate_llm(
    csv_path: str,
    api_key: Optional[str] = None,
    mode: str = "zero_shot",         # zero_shot | few_shot_3 | few_shot_5 | cot
    k_shot: int = 3,
    model: str = "claude-sonnet-4-20250514",
    max_samples: Optional[int] = None,   # None = full test set
    save_dir: str = "results",
    rate_limit_delay: float = 0.5,       # Seconds between API calls
) -> dict:
    """
    Run LLM-based evaluation on BanMANI test set.
    """
    if not ANTHROPIC_AVAILABLE:
        print("❌ anthropic package not available.")
        return {}

    api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not set.")
        return {}

    client = anthropic.Anthropic(api_key=api_key)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  LLM Evaluation: {mode.upper()}  |  Model: {model}")
    print(f"{'='*60}")

    # Load data
    train_df, test_df = load_dataset(csv_path)
    if max_samples:
        test_df = test_df.sample(min(max_samples, len(test_df)), random_state=42)
        print(f"   Using {len(test_df)} test samples (limited for cost)")

    # Prepare few-shot examples
    examples = []
    if 'few_shot' in mode:
        examples = select_few_shot_examples(train_df, k=k_shot, strategy='balanced')
        print(f"   Selected {len(examples)} few-shot examples")

    # ── Inference Loop ────────────────────────────────────────────────────────
    predictions = []
    raw_responses = []

    for i, row in test_df.iterrows():
        claim   = row['mani_news'] or ""
        article = (row['original_news_article'] or "")[:1500]  # Truncate for token budget

        # Build prompt
        if mode == "zero_shot":
            prompt = ZERO_SHOT_TEMPLATE.format(claim=claim, article=article)
        elif mode == "cot":
            prompt = COT_TEMPLATE.format(claim=claim, article=article)
        elif 'few_shot' in mode:
            ex_text = format_examples(examples)
            prompt  = FEW_SHOT_TEMPLATE.format(
                k=len(examples), examples=ex_text,
                claim=claim, article=article
            )
        else:
            prompt = ZERO_SHOT_TEMPLATE.format(claim=claim, article=article)

        response_text = call_claude(client, prompt, model=model)
        parsed        = parse_response(response_text)

        predictions.append(parsed)
        raw_responses.append({'index': i, 'response': response_text, 'parsed': parsed})

        if (len(predictions)) % 10 == 0:
            print(f"   Processed {len(predictions)}/{len(test_df)} samples...")

        time.sleep(rate_limit_delay)

    # ── Metrics ───────────────────────────────────────────────────────────────
    y_true  = test_df['label'].tolist()
    y_pred  = [1 if p['label'] == 'MANI' else 0 for p in predictions]
    y_conf  = [p['confidence'] for p in predictions]

    acc    = accuracy_score(y_true, y_pred)
    f1_m   = f1_score(y_true, y_pred, average='macro')
    f1_w   = f1_score(y_true, y_pred, average='weighted')
    f1_mani = f1_score(y_true, y_pred, pos_label=1)
    cm     = confusion_matrix(y_true, y_pred)

    print(f"\n{'='*50}")
    print(f"  Results: {mode.upper()}")
    print(f"{'='*50}")
    print(f"  Accuracy        : {acc:.4f}")
    print(f"  F1 (macro)      : {f1_m:.4f}")
    print(f"  F1 (weighted)   : {f1_w:.4f}")
    print(f"  F1 (MANI class) : {f1_mani:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"             Pred NO_MANI  Pred MANI")
    print(f"  True NO_MANI  {cm[0,0]:>6}       {cm[0,1]:>6}")
    print(f"  True MANI     {cm[1,0]:>6}       {cm[1,1]:>6}")
    print()
    print(classification_report(y_true, y_pred, target_names=['NO_MANI', 'MANI']))

    # ── Error Analysis ────────────────────────────────────────────────────────
    errors = []
    for i, (true, pred, row) in enumerate(zip(y_true, y_pred, test_df.itertuples())):
        if true != pred:
            errors.append({
                'index': i,
                'true_label': 'MANI' if true == 1 else 'NO_MANI',
                'pred_label': 'MANI' if pred == 1 else 'NO_MANI',
                'confidence': y_conf[i],
                'category': row.category,
                'claim': row.mani_news[:100],
                'reason': predictions[i].get('reason', ''),
            })

    print(f"\n  Error Analysis: {len(errors)} errors")
    print(f"  False Positives (NO_MANI predicted as MANI): {sum(1 for e in errors if e['true_label']=='NO_MANI')}")
    print(f"  False Negatives (MANI predicted as NO_MANI): {sum(1 for e in errors if e['true_label']=='MANI')}")

    # Error by category
    if errors:
        by_cat = {}
        for e in errors:
            by_cat[e['category']] = by_cat.get(e['category'], 0) + 1
        print(f"\n  Errors by category: {dict(sorted(by_cat.items(), key=lambda x: -x[1]))}")

    # Save results
    results = {
        'mode': mode,
        'model': model,
        'n_samples': len(test_df),
        'metrics': {
            'accuracy': round(acc, 4),
            'f1_macro': round(f1_m, 4),
            'f1_weighted': round(f1_w, 4),
            'f1_mani': round(f1_mani, 4),
            'confusion_matrix': cm.tolist(),
        },
        'error_analysis': {
            'total_errors': len(errors),
            'false_positives': sum(1 for e in errors if e['true_label']=='NO_MANI'),
            'false_negatives': sum(1 for e in errors if e['true_label']=='MANI'),
            'errors': errors[:20],  # Save first 20 for inspection
        },
    }

    results_path = os.path.join(save_dir, f'llm_{mode}_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Results saved to {results_path}")

    # Save raw responses
    raw_path = os.path.join(save_dir, f'llm_{mode}_raw.json')
    with open(raw_path, 'w', encoding='utf-8') as f:
        json.dump(raw_responses, f, ensure_ascii=False, indent=2)

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "../data/BanMANI.csv"
    mode     = sys.argv[2] if len(sys.argv) > 2 else "zero_shot"

    results = evaluate_llm(
        csv_path=csv_path,
        mode=mode,
        max_samples=50,       # Start small for cost estimation
    )
