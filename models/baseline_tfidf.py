"""
Baseline Model: TF-IDF + Logistic Regression (and SVM)
Fast, interpretable — sets the performance floor for Bangla manipulation detection.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
import joblib
import json
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data.preprocess import load_dataset, get_class_weights


# ── TF-IDF Configuration ────────────────────────────────────────────────────
# Bangla text works well with character n-grams (morphologically rich)
# Word-level also helpful; we combine both.

CHAR_TFIDF_CONFIG = dict(
    analyzer='char_wb',
    ngram_range=(2, 5),
    max_features=100_000,
    sublinear_tf=True,
    min_df=2,
    strip_accents=None,   # Don't strip — Bangla diacritics carry meaning
)

WORD_TFIDF_CONFIG = dict(
    analyzer='word',
    ngram_range=(1, 3),
    max_features=80_000,
    sublinear_tf=True,
    min_df=2,
    strip_accents=None,
)


def build_tfidf_lr_pipeline(use_char_ngrams: bool = True) -> Pipeline:
    """
    TF-IDF (char n-grams) + Logistic Regression pipeline.
    """
    config = CHAR_TFIDF_CONFIG if use_char_ngrams else WORD_TFIDF_CONFIG
    return Pipeline([
        ('tfidf', TfidfVectorizer(**config)),
        ('clf',   LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight='balanced',
            solver='lbfgs',
            random_state=42,
        ))
    ])


def build_tfidf_svm_pipeline() -> Pipeline:
    """
    TF-IDF (word n-grams) + LinearSVC pipeline.
    """
    return Pipeline([
        ('tfidf', TfidfVectorizer(**WORD_TFIDF_CONFIG)),
        ('clf',   LinearSVC(
            C=0.5,
            max_iter=2000,
            class_weight='balanced',
            random_state=42,
        ))
    ])


def evaluate(y_true, y_pred, model_name: str = "Model", y_prob=None) -> dict:
    """
    Compute and print full evaluation metrics.
    Returns results dict.
    """
    acc  = accuracy_score(y_true, y_pred)
    f1_m = f1_score(y_true, y_pred, average='macro')
    f1_w = f1_score(y_true, y_pred, average='weighted')
    f1_mani = f1_score(y_true, y_pred, pos_label=1)
    cm   = confusion_matrix(y_true, y_pred)
    auc  = roc_auc_score(y_true, y_prob) if y_prob is not None else None

    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")
    print(f"  Accuracy        : {acc:.4f}")
    print(f"  F1 (macro)      : {f1_m:.4f}")
    print(f"  F1 (weighted)   : {f1_w:.4f}")
    print(f"  F1 (MANI class) : {f1_mani:.4f}")
    if auc:
        print(f"  ROC-AUC         : {auc:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"             Pred NO_MANI  Pred MANI")
    print(f"  True NO_MANI  {cm[0,0]:>6}       {cm[0,1]:>6}")
    print(f"  True MANI     {cm[1,0]:>6}       {cm[1,1]:>6}")
    print()
    print(classification_report(y_true, y_pred, target_names=['NO_MANI', 'MANI']))

    return {
        'model': model_name,
        'accuracy': round(acc, 4),
        'f1_macro': round(f1_m, 4),
        'f1_weighted': round(f1_w, 4),
        'f1_mani': round(f1_mani, 4),
        'roc_auc': round(auc, 4) if auc else None,
        'confusion_matrix': cm.tolist(),
    }


def cross_validate_baseline(pipeline, X_train, y_train, cv=5) -> dict:
    """5-fold stratified cross-validation on training set."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X_train, y_train,
                             cv=skf, scoring='f1_macro', n_jobs=-1)
    print(f"  CV F1 (macro) per fold: {np.round(scores, 4)}")
    print(f"  Mean: {scores.mean():.4f}  Std: {scores.std():.4f}")
    return {'cv_f1_mean': round(scores.mean(), 4), 'cv_f1_std': round(scores.std(), 4)}


def get_top_features(pipeline, top_n=20) -> tuple:
    """Extract top features for interpretability."""
    tfidf = pipeline.named_steps['tfidf']
    clf   = pipeline.named_steps['clf']
    feature_names = tfidf.get_feature_names_out()

    if hasattr(clf, 'coef_'):
        coef = clf.coef_[0] if clf.coef_.ndim > 1 else clf.coef_
        top_mani    = np.argsort(coef)[-top_n:][::-1]
        top_nomani  = np.argsort(coef)[:top_n]
        return (
            [(feature_names[i], round(float(coef[i]), 4)) for i in top_mani],
            [(feature_names[i], round(float(coef[i]), 4)) for i in top_nomani],
        )
    return [], []


def run_baseline(csv_path: str, text_col: str = 'text_concat',
                 save_dir: str = 'results') -> dict:
    """
    Full baseline pipeline: load → train → evaluate → save.
    """
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n🔵 Loading BanMANI dataset from: {csv_path}")
    train_df, test_df = load_dataset(csv_path)

    X_train = train_df[text_col]
    y_train = train_df['label']
    X_test  = test_df[text_col]
    y_test  = test_df['label']

    all_results = {}

    # ── 1. TF-IDF Char n-gram + LR ───────────────────────────────────────────
    print("\n🔵 Training: TF-IDF (char 2-5gram) + Logistic Regression")
    pipe_lr = build_tfidf_lr_pipeline(use_char_ngrams=True)
    cv_res = cross_validate_baseline(pipe_lr, X_train, y_train)
    pipe_lr.fit(X_train, y_train)
    y_pred = pipe_lr.predict(X_test)
    try:
        y_prob = pipe_lr.predict_proba(X_test)[:, 1]
    except Exception:
        y_prob = None
    res = evaluate(y_test, y_pred, "TF-IDF (char) + LR", y_prob)
    res.update(cv_res)
    all_results['tfidf_char_lr'] = res

    # Feature importance
    top_mani, top_nomani = get_top_features(pipe_lr)
    if top_mani:
        print("\n  Top 10 MANI-predictive n-grams:")
        for feat, w in top_mani[:10]:
            print(f"    {feat!r:30s} → {w}")
        all_results['tfidf_char_lr']['top_mani_features'] = top_mani[:10]

    # ── 2. TF-IDF Word n-gram + LR ───────────────────────────────────────────
    print("\n🔵 Training: TF-IDF (word 1-3gram) + Logistic Regression")
    pipe_word_lr = Pipeline([
        ('tfidf', TfidfVectorizer(**WORD_TFIDF_CONFIG)),
        ('clf',   LogisticRegression(C=1.0, max_iter=1000,
                                     class_weight='balanced',
                                     solver='lbfgs', random_state=42))
    ])
    cross_validate_baseline(pipe_word_lr, X_train, y_train)
    pipe_word_lr.fit(X_train, y_train)
    y_pred_w = pipe_word_lr.predict(X_test)
    try:
        y_prob_w = pipe_word_lr.predict_proba(X_test)[:, 1]
    except Exception:
        y_prob_w = None
    res_w = evaluate(y_test, y_pred_w, "TF-IDF (word) + LR", y_prob_w)
    all_results['tfidf_word_lr'] = res_w

    # ── 3. TF-IDF + LinearSVC ────────────────────────────────────────────────
    print("\n🔵 Training: TF-IDF (word) + LinearSVC")
    pipe_svm = build_tfidf_svm_pipeline()
    cross_validate_baseline(pipe_svm, X_train, y_train)
    pipe_svm.fit(X_train, y_train)
    y_pred_svm = pipe_svm.predict(X_test)
    res_svm = evaluate(y_test, y_pred_svm, "TF-IDF + LinearSVC")
    all_results['tfidf_svm'] = res_svm

    # ── Save best model ───────────────────────────────────────────────────────
    best_key = max(all_results, key=lambda k: all_results[k]['f1_macro'])
    best_pipe = {'tfidf_char_lr': pipe_lr, 'tfidf_word_lr': pipe_word_lr,
                 'tfidf_svm': pipe_svm}[best_key]
    model_path = os.path.join(save_dir, 'best_baseline.joblib')
    joblib.dump(best_pipe, model_path)
    print(f"\n✅ Best baseline: {best_key} → saved to {model_path}")

    # Save metrics
    results_path = os.path.join(save_dir, 'baseline_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"✅ Results saved to {results_path}")

    return all_results


if __name__ == "__main__":
    csv = sys.argv[1] if len(sys.argv) > 1 else "../data/BanMANI.csv"
    results = run_baseline(csv)
