"""
High-Performance Parallel Master Experiment Runner:
- 30 Independent Randomized Repeats of 10-Fold Stratified Cross-Validation (300 evaluations per algorithm)
- Multiprocessing across 12-16 CPU cores via joblib
- Comparison of Classification Algorithms:
    1. Logistic Regression (Primary baseline from Barak Or 2025)
    2. Linear Support Vector Machine (Linear SVM)
    3. Random Forest (RF)
- Generalization testing on FNFC (7,034 samples) and nfr (625 samples)
- Full Statistical Validation for Category W (t-test, Wilcoxon, Cohen's d, ANOVA)
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from collections import Counter
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.dataset_loader import load_promise_exp, load_fnfc, load_nfr
from models.pipeline import HybridFeatureExtractor
from neuro.generator import NeuroSymbolicAugmenter
from neuro.boundary_cleaner import DualSpaceBoundaryCleaner
from evaluation.metrics import calculate_comprehensive_metrics, compute_statistical_tests

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek


def fit_and_evaluate_single_fold(
    train_texts, train_labels,
    test_texts, test_labels,
    method, clf_name, random_state
):
    """Executes a single fold pipeline and returns evaluation metrics."""
    # 1. Resampling
    if method == 'raw':
        final_train_texts = list(train_texts)
        final_train_labels = list(train_labels)
        extractor = HybridFeatureExtractor(max_features=1200, n_components=40, random_state=random_state)
        X_train = extractor.fit_transform(final_train_texts)
        y_train = np.array(final_train_labels)
        
    elif method == 'barak_or':
        extractor = HybridFeatureExtractor(max_features=1000, n_components=30, random_state=random_state)
        X_raw = extractor.fit_transform(train_texts)
        y_raw = np.array(train_labels)
        
        min_c = min(Counter(y_raw).values())
        k_neighbors = max(1, min(3, min_c - 1))
        smote = SMOTE(k_neighbors=k_neighbors, random_state=random_state)
        tomek = TomekLinks()
        smt = SMOTETomek(smote=smote, tomek=tomek, random_state=random_state)
        try:
            X_train, y_train = smt.fit_resample(X_raw, y_raw)
        except Exception:
            X_train, y_train = X_raw, y_raw
            
    elif method == 'ali_et_al':
        extractor = HybridFeatureExtractor(max_features=1200, n_components=40, random_state=random_state)
        X_raw = extractor.fit_transform(train_texts)
        y_raw = np.array(train_labels)
        
        min_c = min(Counter(y_raw).values())
        k_neighbors = max(1, min(3, min_c - 1))
        smote = SMOTE(k_neighbors=k_neighbors, random_state=random_state)
        tomek = TomekLinks()
        smt = SMOTETomek(smote=smote, tomek=tomek, random_state=random_state)
        try:
            X_train, y_train = smt.fit_resample(X_raw, y_raw)
        except Exception:
            X_train, y_train = X_raw, y_raw
            
    elif method == 'proposed':
        augmenter = NeuroSymbolicAugmenter(random_seed=random_state)
        cleaner = DualSpaceBoundaryCleaner(random_state=random_state)
        
        counts = Counter(train_labels)
        target_count = max(counts.values()) // 2
        
        augmented_texts = list(train_texts)
        augmented_labels = list(train_labels)
        
        for cls, count in counts.items():
            if cls == 'F':
                continue
            if count < target_count:
                needed = min(target_count - count, 35)
                gen_texts, _ = augmenter.generate_verified_batch(cls, needed)
                augmented_texts.extend(gen_texts)
                augmented_labels.extend([cls] * len(gen_texts))
                
        cleaned_texts, cleaned_labels, _ = cleaner.fit_and_clean(augmented_texts, augmented_labels)
        extractor = HybridFeatureExtractor(max_features=1200, n_components=40, random_state=random_state)
        X_train = extractor.fit_transform(cleaned_texts)
        y_train = np.array(cleaned_labels)
    else:
        raise ValueError(f"Unknown method {method}")
        
    X_test = extractor.transform(test_texts)
    y_test = np.array(test_labels)
    
    # 2. Classifier
    if clf_name == 'logistic_regression':
        clf = LogisticRegression(max_iter=500, random_state=random_state, C=1.5)
    elif clf_name == 'linear_svm':
        clf = LinearSVC(random_state=random_state, C=1.0, max_iter=1000)
    elif clf_name == 'random_forest':
        clf = RandomForestClassifier(n_estimators=80, random_state=random_state, n_jobs=1)
    else:
        clf = LogisticRegression(max_iter=500, random_state=random_state)
        
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    unique_labels = sorted(list(set(test_labels)))
    metrics = calculate_comprehensive_metrics(y_test, y_pred, labels=unique_labels)
    return metrics


def run_full_parallel_benchmark(n_repeats: int = 30, n_splits: int = 10, n_jobs: int = 12):
    """Runs 30 randomized splits of 10-fold CV in parallel."""
    print("=" * 75)
    print(f"STARTING PARALLEL BENCHMARK: {n_repeats} REPEATS x {n_splits} FOLDS ({n_repeats * n_splits} TOTAL RUNS)")
    print(f"UTILIZING {n_jobs} PARALLEL CPU CORES")
    print("=" * 75)
    
    df_promise = load_promise_exp()
    texts = df_promise['text'].tolist()
    labels = df_promise['label'].tolist()
    
    methods = ['raw', 'barak_or', 'ali_et_al', 'proposed']
    classifiers = ['logistic_regression', 'linear_svm', 'random_forest']
    
    results = {
        'metadata': {
            'n_repeats': n_repeats,
            'n_splits': n_splits,
            'classifiers': classifiers,
            'methods': methods,
            'cores': n_jobs
        },
        'cv_metrics': {},
        'generalization': {},
        'statistical_tests': {}
    }
    
    start_time = time.time()
    
    # For each classifier, benchmark all methods
    for clf in classifiers:
        print(f"\n>>> BENCHMARKING CLASSIFIER: {clf.upper()} <<<")
        results['cv_metrics'][clf] = {}
        
        # Prepare all fold tasks
        tasks = []
        for repeat in range(n_repeats):
            seed = 1000 + repeat * 47
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            for fold, (train_idx, test_idx) in enumerate(skf.split(texts, labels)):
                fold_train_t = [texts[i] for i in train_idx]
                fold_train_l = [labels[i] for i in train_idx]
                fold_test_t = [texts[i] for i in test_idx]
                fold_test_l = [labels[i] for i in test_idx]
                
                for m in methods:
                    tasks.append((repeat, fold, m, clf, fold_train_t, fold_train_l, fold_test_t, fold_test_l, seed + fold))
                    
        print(f"Generated {len(tasks)} parallel fold tasks. Executing across CPU cluster...")
        
        # Run parallel execution with joblib
        parallel_outputs = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(fit_and_evaluate_single_fold)(
                task[4], task[5], task[6], task[7], task[2], task[3], task[8]
            ) for task in tasks
        )
        
        # Aggregate results per method
        method_runs = {m: {'accuracy': [], 'f1_score': [], 'precision': [], 'recall': [], 'specificity': [], 'mcc': []} for m in methods}
        
        for task, metrics in zip(tasks, parallel_outputs):
            m = task[2]
            for metric_name in method_runs[m].keys():
                method_runs[m][metric_name].append(metrics[metric_name])
                
        # Calculate summary across the 30 repeats (mean per repeat, then overall mean & std)
        for m in methods:
            results['cv_metrics'][clf][m] = {}
            for metric_name, all_fold_values in method_runs[m].items():
                # Group by repeat (each repeat has n_splits folds)
                repeat_means = [float(np.mean(all_fold_values[r * n_splits : (r + 1) * n_splits])) for r in range(n_repeats)]
                results['cv_metrics'][clf][m][metric_name] = {
                    'mean': float(np.mean(repeat_means)),
                    'std': float(np.std(repeat_means)),
                    '30_repeat_means': repeat_means
                }
                
            mean_acc = results['cv_metrics'][clf][m]['accuracy']['mean'] * 100
            std_acc = results['cv_metrics'][clf][m]['accuracy']['std'] * 100
            mean_f1 = results['cv_metrics'][clf][m]['f1_score']['mean'] * 100
            print(f"  [{m:<12}] Acc: {mean_acc:.2f}% ± {std_acc:.2f}% | F1: {mean_f1:.2f}%")
            
        # Statistical Significance Tests for this classifier (Proposed vs. others across 30 repeats)
        results['statistical_tests'][clf] = {}
        prop_accs = results['cv_metrics'][clf]['proposed']['accuracy']['30_repeat_means']
        for base in ['raw', 'barak_or', 'ali_et_al']:
            base_accs = results['cv_metrics'][clf][base]['accuracy']['30_repeat_means']
            results['statistical_tests'][clf][f'proposed_vs_{base}'] = compute_statistical_tests(prop_accs, base_accs)
            
    # Generalization test on full datasets
    print("\n" + "=" * 75)
    print("RUNNING ZERO-SHOT CROSS-DATASET GENERALIZATION ON FNFC & NFR")
    print("=" * 75)
    df_fnfc = load_fnfc()
    df_nfr = load_nfr()
    
    train_texts = df_promise['text'].tolist()
    train_labels = df_promise['label'].tolist()
    
    for clf in ['logistic_regression', 'linear_svm']:
        results['generalization'][clf] = {'FNFC': {}, 'nfr': {}}
        for m in methods:
            # FNFC
            m_fnfc = fit_and_evaluate_single_fold(
                train_texts, train_labels,
                df_fnfc['text'].tolist(), df_fnfc['label'].tolist(),
                m, clf, 42
            )
            results['generalization'][clf]['FNFC'][m] = m_fnfc
            
            # nfr
            m_nfr = fit_and_evaluate_single_fold(
                train_texts, train_labels,
                df_nfr['text'].tolist(), df_nfr['label'].tolist(),
                m, clf, 42
            )
            results['generalization'][clf]['nfr'][m] = m_nfr
            print(f"[{clf} | {m:<10}] Transfer FNFC Acc: {m_fnfc['accuracy']*100:.2f}% | Transfer nfr Acc: {m_nfr['accuracy']*100:.2f}%")

    elapsed = time.time() - start_time
    results['metadata']['elapsed_seconds'] = elapsed
    print(f"\nExecution finished in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes).")
    
    out_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rigorous_30_split_results.json')
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"Saved complete publication results to: {out_file}")
    return results


if __name__ == '__main__':
    run_full_parallel_benchmark(n_repeats=30, n_splits=10, n_jobs=12)
