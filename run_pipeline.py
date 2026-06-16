#!/usr/bin/env python3
"""
BanMANI Research Pipeline — Main Entry Point
Usage:
    python run_pipeline.py --stage baseline
    python run_pipeline.py --stage llm --mode zero_shot
    python run_pipeline.py --stage llm --mode few_shot_3
    python run_pipeline.py --stage transformer --model xlmr
    python run_pipeline.py --stage all
"""

import argparse
import os
import sys
import shutil

# ── Path Setup ────────────────────────────────────────────────────────────────
ROOT    = os.path.dirname(os.path.abspath(__file__))
CSV     = os.path.join(ROOT, 'data', 'BanMANI.csv')
RESULTS = os.path.join(ROOT, 'results', 'outputs')
os.makedirs(RESULTS, exist_ok=True)
sys.path.insert(0, ROOT)


def stage_baseline(args):
    from models.baseline_tfidf import run_baseline
    print("\n🟦 STAGE: TF-IDF Baseline")
    run_baseline(
        csv_path=CSV,
        text_col='text_concat',   # Word-level input for TF-IDF
        save_dir=RESULTS,
    )


def stage_llm(args):
    from models.llm_evaluation import evaluate_llm
    print(f"\n🟨 STAGE: LLM Evaluation [{args.mode}]")
    evaluate_llm(
        csv_path=CSV,
        mode=args.mode,
        k_shot=int(args.mode.split('_')[-1]) if 'few_shot' in args.mode else 3,
        max_samples=args.max_samples,
        save_dir=RESULTS,
    )


def stage_transformer(args):
    from models.transformer_finetune import fine_tune_transformer, FineTuneConfig
    print(f"\n🟩 STAGE: Transformer Fine-tuning [{args.model}]")
    cfg = FineTuneConfig(
        model_key=args.model,
        text_col='text_sep_token',
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        save_dir=os.path.join(RESULTS, 'transformer'),
    )
    fine_tune_transformer(csv_path=CSV, config=cfg)


def stage_analysis(args):
    from results.analysis import run_full_analysis
    print("\n🟥 STAGE: Analysis & Comparison")
    run_full_analysis(csv_path=CSV, results_dir=RESULTS)


# ── Argument Parser ───────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='BanMANI Manipulation Detection Pipeline'
    )
    parser.add_argument('--stage', type=str, default='baseline',
                        choices=['baseline', 'llm', 'transformer', 'analysis', 'all'],
                        help='Which stage to run')
    parser.add_argument('--mode', type=str, default='zero_shot',
                        choices=['zero_shot', 'few_shot_3', 'few_shot_5', 'cot'],
                        help='LLM evaluation mode')
    parser.add_argument('--model', type=str, default='xlmr',
                        choices=['mbert', 'xlmr', 'banglabert', 'xlmr_large'],
                        help='Transformer model key')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Training epochs for transformer')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for transformer training')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Limit test samples for LLM evaluation (cost control)')
    parser.add_argument('--csv', type=str, default=None,
                        help='Path to BanMANI.csv (overrides default)')

    args = parser.parse_args()

    # Override CSV path if provided
    global CSV
    if args.csv:
        CSV = args.csv

    if not os.path.exists(CSV):
        print(f"❌ Dataset not found at: {CSV}")
        print("   Place BanMANI.csv in banmani_research/data/ or use --csv <path>")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  BanMANI Research Pipeline")
    print(f"  Dataset: {CSV}")
    print(f"  Results: {RESULTS}")
    print(f"{'='*60}")

    if args.stage == 'all':
        stage_baseline(args)
        stage_llm(args)
        stage_analysis(args)
    elif args.stage == 'baseline':
        stage_baseline(args)
    elif args.stage == 'llm':
        stage_llm(args)
    elif args.stage == 'transformer':
        stage_transformer(args)
    elif args.stage == 'analysis':
        stage_analysis(args)

    print(f"\n✅ Pipeline stage '{args.stage}' complete.")
    print(f"   Results saved in: {RESULTS}")


if __name__ == "__main__":
    main()
