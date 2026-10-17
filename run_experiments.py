"""
Master Experiment Runner:
1. 10-Fold Stratified Cross-Validation on Promise_Exp
2. Cross-Dataset Generalization on FNFC (7,060 samples) and nfr (625 samples)
3. Full Statistical Hypothesis Testing (t-test, Cohen's d, Wilcoxon, ANOVA)
4. Exporting publication-ready metrics for Category W Paper.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold

# Ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.dataset_loader import load_promise_exp, load_fnfc, load_nfr
from models.pipeline import run_pipeline_experiment
from evaluation.metrics import calculate_comprehensive_metrics, compute_statistical_tests

METHODS = ['raw', 'barak_or', 'ali_et_al', 'proposed']
METHOD_LABELS = {
    'raw': 'Raw Baseline (No Resampling)',
    'barak_or': 'Barak Or (2025) [SMOTE-Tomek]',
    'ali_et_al': 'Ali et al. (2026) [Hybrid TF-IDF+SBERT + SMOTE-Tomek]',
    'proposed': 'Proposed [Explainable Neuro-Symbolic ISO 25010]'
}


def run_cross_validation_experiments(n_splits: int = 10, random_state: int = 42):
    """Runs 10-fold stratified CV on Promise_Exp across all 4 methods."""
    print("=" * 70)
    print("1. LOADING TRAINING DATASET (Promise_Exp.csv)")
    print("=" * 70)
    df_promise = load_promise_exp()
    print(f"Total samples: {len(df_promise)} across {df_promise['label'].nunique()} classes.")
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    cv_results = defaultdict(lambda: defaultdict(list))
    
    print("\n" + "=" * 70)
    print(f"2. EXECUTING {n_splits}-FOLD STRATIFIED CROSS-VALIDATION")
    print("=" * 70)
    
    unique_labels = sorted(df_promise['label'].unique())
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(df_promise['text'], df_promise['label']), 1):
        print(f"\n--- Fold {fold}/{n_splits} ---")
        train_texts = df_promise['text'].iloc[train_idx].tolist()
        train_labels = df_promise['label'].iloc[train_idx].tolist()
        test_texts = df_promise['text'].iloc[test_idx].tolist()
        test_labels = df_promise['label'].iloc[test_idx].tolist()
        
        for m in METHODS:
            res = run_pipeline_experiment(
                train_texts=train_texts,
                train_labels=train_labels,
                test_texts=test_texts,
                test_labels=test_labels,
                method=m,
                classifier_name='logistic_regression',
                random_state=random_state + fold
            )
            metrics = calculate_comprehensive_metrics(res['y_true'], res['y_pred'], labels=unique_labels)
            for k, val in metrics.items():
                cv_results[m][k].append(val)
                
            print(f"  [{m:<12}] Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f} | MCC: {metrics['mcc']:.4f}")
            
    # Compute summary means and stds
    summary_cv = {}
    for m in METHODS:
        summary_cv[m] = {}
        for metric, vals in cv_results[m].items():
            summary_cv[m][metric] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
                'raw_folds': [float(v) for v in vals]
            }
            
    return summary_cv, df_promise


def run_generalization_experiments(df_promise: pd.DataFrame, random_state: int = 42):
    """
    Trains models on full Promise_Exp and tests zero-shot cross-dataset generalization
    on FNFC (7,060 samples) and nfr (625 samples).
    """
    print("\n" + "=" * 70)
    print("3. EXECUTING CROSS-DATASET GENERALIZATION EVALUATION")
    print("=" * 70)
    
    df_fnfc = load_fnfc()
    df_nfr = load_nfr()
    print(f"Generalization Target 1 (FNFC): {len(df_fnfc)} samples")
    print(f"Generalization Target 2 (nfr):  {len(df_nfr)} samples")
    
    unique_labels = sorted(df_promise['label'].unique())
    train_texts = df_promise['text'].tolist()
    train_labels = df_promise['label'].tolist()
    
    gen_results = {'FNFC': {}, 'nfr': {}}
    
    for m in METHODS:
        print(f"\nTraining on Promise_Exp with [{m}]...")
        # Evaluate on FNFC
        res_fnfc = run_pipeline_experiment(
            train_texts=train_texts,
            train_labels=train_labels,
            test_texts=df_fnfc['text'].tolist(),
            test_labels=df_fnfc['label'].tolist(),
            method=m,
            classifier_name='logistic_regression',
            random_state=random_state
        )
        metrics_fnfc = calculate_comprehensive_metrics(res_fnfc['y_true'], res_fnfc['y_pred'], labels=unique_labels)
        gen_results['FNFC'][m] = metrics_fnfc
        print(f"  -> Transfer to FNFC: Acc: {metrics_fnfc['accuracy']:.4f} | F1: {metrics_fnfc['f1_score']:.4f} | MCC: {metrics_fnfc['mcc']:.4f}")
        
        # Evaluate on nfr
        res_nfr = run_pipeline_experiment(
            train_texts=train_texts,
            train_labels=train_labels,
            test_texts=df_nfr['text'].tolist(),
            test_labels=df_nfr['label'].tolist(),
            method=m,
            classifier_name='logistic_regression',
            random_state=random_state
        )
        metrics_nfr = calculate_comprehensive_metrics(res_nfr['y_true'], res_nfr['y_pred'], labels=unique_labels)
        gen_results['nfr'][m] = metrics_nfr
        print(f"  -> Transfer to nfr:  Acc: {metrics_nfr['accuracy']:.4f} | F1: {metrics_nfr['f1_score']:.4f} | MCC: {metrics_nfr['mcc']:.4f}")
        
    return gen_results


def main():
    cv_summary, df_promise = run_cross_validation_experiments(n_splits=10)
    gen_summary = run_generalization_experiments(df_promise)
    
    # Statistical tests between Proposed vs Baselines
    stat_tests = {}
    proposed_accs = cv_summary['proposed']['accuracy']['raw_folds']
    for b in ['raw', 'barak_or', 'ali_et_al']:
        base_accs = cv_summary[b]['accuracy']['raw_folds']
        stat_tests[f'proposed_vs_{b}'] = compute_statistical_tests(proposed_accs, base_accs)
        
    final_output = {
        'cross_validation': cv_summary,
        'generalization': gen_summary,
        'statistical_tests': stat_tests
    }
    
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experimental_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2)
        
    print("\n" + "=" * 70)
    print(f"ALL EXPERIMENTS COMPLETED SUCCESSFULLY! Results saved to: {out_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()
