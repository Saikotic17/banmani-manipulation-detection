# %%
import sys, os
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_predict
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

from data.preprocess import load_dataset

train_df, test_df = load_dataset('data/BanMANI.csv')
print(f"Train: {len(train_df)} ({train_df['label'].mean()*100:.1f}% MANI)")
print(f"Test:  {len(test_df)} ({test_df['label'].mean()*100:.1f}% MANI)")

WORD_CFG = dict(analyzer='word', ngram_range=(1,3), max_features=80000,
                sublinear_tf=True, min_df=2, strip_accents=None)

def make_pipeline():
    return Pipeline([
        ('tfidf', TfidfVectorizer(**WORD_CFG)),
        ('clf', LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced',
                                    solver='lbfgs', random_state=42))
    ])
# %%
# %%
import sys, os
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_predict
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

from data.preprocess import load_dataset

train_df, test_df = load_dataset('data/BanMANI.csv')
print(f"Train: {len(train_df)} ({train_df['label'].mean()*100:.1f}% MANI)")
print(f"Test:  {len(test_df)} ({test_df['label'].mean()*100:.1f}% MANI)")

WORD_CFG = dict(analyzer='word', ngram_range=(1,3), max_features=80000,
                sublinear_tf=True, min_df=2, strip_accents=None)

def make_pipeline():
    return Pipeline([
        ('tfidf', TfidfVectorizer(**WORD_CFG)),
        ('clf', LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced',
                                    solver='lbfgs', random_state=42))
    ])

# %%
pipe = make_pipeline()
pipe.fit(train_df['text_concat'], train_df['label'])
y_pred = pipe.predict(test_df['text_concat'])
f1_original = f1_score(test_df['label'], y_pred, average='macro')
print(f"Original fixed-split F1 Macro: {f1_original:.4f}")
print("(This is your reported baseline: 0.5717)")

# %% OPTION B - Threshold Calibration
# Step 1: Find calibrated threshold using cross-validated training predictions
oof_probs = cross_val_predict(
    make_pipeline(), train_df['text_concat'], train_df['label'],
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    method='predict_proba'
)[:, 1]

thresholds = np.arange(0.10, 0.91, 0.02)
oof_f1s = []
for t in thresholds:
    preds = (oof_probs > t).astype(int)
    oof_f1s.append(f1_score(train_df['label'], preds, average='macro'))

best_idx = np.argmax(oof_f1s)
best_threshold = thresholds[best_idx]
print(f"Default threshold: 0.50")
print(f"Calibrated threshold (from CV on train): {best_threshold:.2f}")
print(f"OOF F1 Macro at calibrated threshold: {oof_f1s[best_idx]:.4f}")
print(f"OOF F1 Macro at default 0.50:          {oof_f1s[np.argmin(np.abs(thresholds-0.5))]:.4f}")

# %%
# Step 2: Apply calibrated threshold to test set, compare
pipe_final = make_pipeline()
pipe_final.fit(train_df['text_concat'], train_df['label'])
test_probs = pipe_final.predict_proba(test_df['text_concat'])[:, 1]

pred_default = (test_probs > 0.5).astype(int)
f1_default = f1_score(test_df['label'], pred_default, average='macro')

pred_calibrated = (test_probs > best_threshold).astype(int)
f1_calibrated = f1_score(test_df['label'], pred_calibrated, average='macro')

print(f"{'='*50}")
print(f"  THRESHOLD CALIBRATION RESULTS")
print(f"{'='*50}")
print(f"Default (0.50)        -> F1 Macro: {f1_default:.4f}")
print(f"Calibrated ({best_threshold:.2f}) -> F1 Macro: {f1_calibrated:.4f}")
print(f"Difference: {f1_calibrated - f1_default:+.4f}")

print("\nConfusion matrix (default 0.50):")
print(confusion_matrix(test_df['label'], pred_default))
print("\nConfusion matrix (calibrated):")
print(confusion_matrix(test_df['label'], pred_calibrated))

# %% OPTION D - Repeated Stratified K-Fold
combined = pd.concat([train_df, test_df], ignore_index=True)
print(f"Combined dataset: {len(combined)} samples ({combined['label'].mean()*100:.1f}% MANI)")

all_f1_scores = []
n_repeats = 10
n_folds = 5

for seed in range(n_repeats):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(combined['text_concat'], combined['label'])):
        tr = combined.iloc[train_idx]
        te = combined.iloc[test_idx]

        pipe = make_pipeline()
        pipe.fit(tr['text_concat'], tr['label'])
        pred = pipe.predict(te['text_concat'])
        f1 = f1_score(te['label'], pred, average='macro')
        all_f1_scores.append(f1)

    print(f"Seed {seed}: {len(all_f1_scores)} folds done, running mean so far: {np.mean(all_f1_scores):.4f}")

all_f1_scores = np.array(all_f1_scores)
print(f"\nTotal runs: {len(all_f1_scores)}")

# %%
mean_f1 = all_f1_scores.mean()
std_f1 = all_f1_scores.std()
se_f1 = std_f1 / np.sqrt(len(all_f1_scores))
ci_95 = 1.96 * se_f1

print(f"{'='*50}")
print(f"  REPEATED STRATIFIED K-FOLD RESULTS (n={len(all_f1_scores)} runs)")
print(f"{'='*50}")
print(f"Mean F1 Macro:        {mean_f1:.4f}")
print(f"Std Dev:              {std_f1:.4f}")
print(f"95% CI:               [{mean_f1-ci_95:.4f}, {mean_f1+ci_95:.4f}]")
print(f"\nOriginal fixed split: {f1_original:.4f}")

if f1_original < mean_f1 - ci_95 or f1_original > mean_f1 + ci_95:
    print(f"\nWARNING: Original fixed-split result ({f1_original:.4f}) falls OUTSIDE the 95% CI")
    print("   -> The original 72.5%/40.7% split was likely an unusually hard test set")
else:
    print(f"\nOK: Original fixed-split result ({f1_original:.4f}) falls WITHIN the 95% CI")
    print("   -> The result is consistent with normal variance, not an artifact of the split")

# %% OPTION A - Repeated Re-Stratified 650/150 Splits
n_seeds = 10
restratified_f1_scores = []

for seed in range(n_seeds):
    tr, te = train_test_split(
        combined, test_size=150, stratify=combined['label'], random_state=seed
    )
    pipe = make_pipeline()
    pipe.fit(tr['text_concat'], tr['label'])
    pred = pipe.predict(te['text_concat'])
    f1 = f1_score(te['label'], pred, average='macro')
    restratified_f1_scores.append(f1)
    print(f"Seed {seed}: train MANI%={tr['label'].mean()*100:.1f}, "
          f"test MANI%={te['label'].mean()*100:.1f}, F1={f1:.4f}")

restratified_f1_scores = np.array(restratified_f1_scores)

# %%
mean_restrat = restratified_f1_scores.mean()
std_restrat = restratified_f1_scores.std()
ci_restrat = 1.96 * std_restrat / np.sqrt(n_seeds)

print(f"{'='*55}")
print(f"  DISTRIBUTION SHIFT TEST")
print(f"{'='*55}")
print(f"Original split (72.5% train / 40.7% test MANI):")
print(f"  F1 Macro = {f1_original:.4f}  (single run)")
print()
print(f"Re-stratified splits (matched train/test distribution):")
print(f"  F1 Macro = {mean_restrat:.4f} +/- {ci_restrat:.4f}  (95% CI, n={n_seeds})")
print()

diff = mean_restrat - f1_original
print(f"Difference: {diff:+.4f}")

if abs(diff) <= ci_restrat:
    print("\n=> Difference is WITHIN the confidence interval.")
    print("   The 72.5%/40.7% distribution mismatch in the original split")
    print("   does NOT appear to be a major driver of the F1 score --")
    print("   the result is consistent with normal sampling variance.")
else:
    print("\n=> Difference EXCEEDS the confidence interval.")
    print("   The distribution mismatch in the original split likely DID")
    print("   meaningfully affect the reported F1 score.")

# %% FINAL SUMMARY
import json

summary = {
    "original_fixed_split": {
        "f1_macro": round(float(f1_original), 4),
        "train_mani_pct": round(float(train_df['label'].mean()*100), 1),
        "test_mani_pct": round(float(test_df['label'].mean()*100), 1),
    },
    "threshold_calibration": {
        "default_threshold_f1": round(float(f1_default), 4),
        "calibrated_threshold": round(float(best_threshold), 2),
        "calibrated_f1": round(float(f1_calibrated), 4),
    },
    "repeated_stratified_kfold": {
        "n_runs": len(all_f1_scores),
        "mean_f1": round(float(mean_f1), 4),
        "std_f1": round(float(std_f1), 4),
        "ci_95_lower": round(float(mean_f1-ci_95), 4),
        "ci_95_upper": round(float(mean_f1+ci_95), 4),
    },
    "restratified_650_150_splits": {
        "n_seeds": n_seeds,
        "mean_f1": round(float(mean_restrat), 4),
        "std_f1": round(float(std_restrat), 4),
        "ci_95_lower": round(float(mean_restrat-ci_restrat), 4),
        "ci_95_upper": round(float(mean_restrat+ci_restrat), 4),
    }
}

print(json.dumps(summary, indent=2))

os.makedirs('results', exist_ok=True)
with open('results/robustness_analysis.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("\nSaved to results/robustness_analysis.json")