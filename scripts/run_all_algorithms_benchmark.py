"""
Unified Multi-Classifier Parallel Benchmark:
Evaluates ALL classification algorithms used across both base papers (Barak Or 2025 & Ali et al. 2026):
1. Logistic Regression (LR)
2. Support Vector Machine (Linear SVM)
3. Random Forest (RF)
4. Multinomial Naive Bayes (MNB)
5. Decision Tree (DT)
6. K-Nearest Neighbors (KNN)

Across all 4 methods:
- Raw Baseline (No Resampling)
- Barak Or (2025) [SMOTE-Tomek]
- Ali et al. (2026) [Hybrid TF-IDF + Dense SBERT with SMOTE-Tomek]
- Proposed [Explainable Neuro-Symbolic ISO/IEC 25010 + Dual-Space Filter]

Evaluated with 30 Independent Randomized Repeats of 10-Fold Stratified Cross-Validation (300 evaluations per algorithm)
plus zero-shot transfer generalization on FNFC (7,034 samples) and nfr (625 samples).
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler

# All 6 Classifiers from the base papers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.dataset_loader import load_promise_exp, load_fnfc, load_nfr
from neuro.generator import NeuroSymbolicAugmenter
from neuro.boundary_cleaner import DualSpaceBoundaryCleaner
from evaluation.metrics import calculate_comprehensive_metrics, compute_statistical_tests

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek

CLASSIFIER_NAMES = [
    'logistic_regression',
    'linear_svm',
    'random_forest',
    'naive_bayes',
    'decision_tree',
    'knn'
]


def evaluate_all_classifiers_on_features(X_train, y_train, X_test, y_test, unique_labels, random_state):
    """Fits and evaluates all 6 classifiers on the identical prepared feature matrices."""
    results = {}
    
    # 1. Logistic Regression
    try:
        lr = LogisticRegression(max_iter=500, random_state=random_state, C=1.5)
        lr.fit(X_train, y_train)
        pred_lr = lr.predict(X_test)
        results['logistic_regression'] = calculate_comprehensive_metrics(y_test, pred_lr, labels=unique_labels)
    except Exception as e:
        results['logistic_regression'] = {'accuracy': 0.0, 'f1_score': 0.0, 'precision': 0.0, 'recall': 0.0, 'specificity': 0.0, 'mcc': 0.0}

    # 2. Linear SVM
    try:
        svm = LinearSVC(random_state=random_state, C=1.0, max_iter=1000)
        svm.fit(X_train, y_train)
        pred_svm = svm.predict(X_test)
        results['linear_svm'] = calculate_comprehensive_metrics(y_test, pred_svm, labels=unique_labels)
    except Exception as e:
        results['linear_svm'] = {'accuracy': 0.0, 'f1_score': 0.0, 'precision': 0.0, 'recall': 0.0, 'specificity': 0.0, 'mcc': 0.0}

    # 3. Random Forest
    try:
        rf = RandomForestClassifier(n_estimators=80, random_state=random_state, n_jobs=1)
        rf.fit(X_train, y_train)
        pred_rf = rf.predict(X_test)
        results['random_forest'] = calculate_comprehensive_metrics(y_test, pred_rf, labels=unique_labels)
    except Exception as e:
        results['random_forest'] = {'accuracy': 0.0, 'f1_score': 0.0, 'precision': 0.0, 'recall': 0.0, 'specificity': 0.0, 'mcc': 0.0}

    # 4. Multinomial Naive Bayes (Requires non-negative inputs)
    try:
        scaler = MinMaxScaler()
        X_train_pos = scaler.fit_transform(X_train)
        X_test_pos = scaler.transform(X_test)
        nb = MultinomialNB()
        nb.fit(X_train_pos, y_train)
        pred_nb = nb.predict(X_test_pos)
        results['naive_bayes'] = calculate_comprehensive_metrics(y_test, pred_nb, labels=unique_labels)
    except Exception as e:
        results['naive_bayes'] = {'accuracy': 0.0, 'f1_score': 0.0, 'precision': 0.0, 'recall': 0.0, 'specificity': 0.0, 'mcc': 0.0}

    # 5. Decision Tree
    try:
        dt = DecisionTreeClassifier(random_state=random_state, max_depth=20)
        dt.fit(X_train, y_train)
        pred_dt = dt.predict(X_test)
        results['decision_tree'] = calculate_comprehensive_metrics(y_test, pred_dt, labels=unique_labels)
    except Exception as e:
        results['decision_tree'] = {'accuracy': 0.0, 'f1_score': 0.0, 'precision': 0.0, 'recall': 0.0, 'specificity': 0.0, 'mcc': 0.0}

    # 6. K-Nearest Neighbors
    try:
        knn = KNeighborsClassifier(n_neighbors=5, n_jobs=1)
        knn.fit(X_train, y_train)
        pred_knn = knn.predict(X_test)
        results['knn'] = calculate_comprehensive_metrics(y_test, pred_knn, labels=unique_labels)
    except Exception as e:
        results['knn'] = {'accuracy': 0.0, 'f1_score': 0.0, 'precision': 0.0, 'recall': 0.0, 'specificity': 0.0, 'mcc': 0.0}

    return results


def process_single_fold_all_classifiers(
    method: str,
    train_texts: list,
    train_labels: list,
    test_texts: list,
    test_labels: list,
    unique_labels: list,
    random_state: int
):
    """Resamples fold according to method, extracts hybrid features, and evaluates ALL 6 classifiers."""
    # 1. Resampling
    if method == 'raw':
        final_train_texts = list(train_texts)
        final_train_labels = list(train_labels)
        
        tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), stop_words='english')
        svd = TruncatedSVD(n_components=35, random_state=random_state)
        
        tf_tr = tfidf.fit_transform(final_train_texts)
        svd_tr = svd.fit_transform(tf_tr)
        X_train = np.hstack([tf_tr.toarray(), svd_tr])
        y_train = np.array(final_train_labels)
        
        tf_te = tfidf.transform(test_texts)
        svd_te = svd.transform(tf_te)
        X_test = np.hstack([tf_te.toarray(), svd_te])
        y_test = np.array(test_labels)

    elif method == 'barak_or':
        tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
        X_raw = tfidf.fit_transform(train_texts).toarray()
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
            
        X_test = tfidf.transform(test_texts).toarray()
        y_test = np.array(test_labels)

    elif method == 'ali_et_al':
        tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), stop_words='english')
        svd = TruncatedSVD(n_components=35, random_state=random_state)
        
        tf_tr = tfidf.fit_transform(train_texts)
        svd_tr = svd.fit_transform(tf_tr)
        X_raw = np.hstack([tf_tr.toarray(), svd_tr])
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
            
        tf_te = tfidf.transform(test_texts)
        svd_te = svd.transform(tf_te)
        X_test = np.hstack([tf_te.toarray(), svd_te])
        y_test = np.array(test_labels)

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
        
        tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), stop_words='english')
        svd = TruncatedSVD(n_components=35, random_state=random_state)
        
        tf_tr = tfidf.fit_transform(cleaned_texts)
        svd_tr = svd.fit_transform(tf_tr)
        X_train = np.hstack([tf_tr.toarray(), svd_tr])
        y_train = np.array(cleaned_labels)
        
        tf_te = tfidf.transform(test_texts)
        svd_te = svd.transform(tf_te)
        X_test = np.hstack([tf_te.toarray(), svd_te])
        y_test = np.array(test_labels)
    else:
        raise ValueError(f"Unknown method {method}")
        
    return evaluate_all_classifiers_on_features(X_train, y_train, X_test, y_test, unique_labels, random_state)


def run_full_suite(n_repeats=30, n_splits=10, n_jobs=12):
    print("=" * 75)
    print("RUNNING ALL 6 CLASSIFIERS FROM BOTH BASE PAPERS:")
    print("1. Logistic Regression (LR)  2. Linear SVM  3. Random Forest (RF)")
    print("4. Multinomial Naive Bayes   5. Decision Tree (DT)  6. KNN")
    print(f"Across 30 Repeats x 10 Folds = 300 Runs per algorithm, in parallel on {n_jobs} cores.")
    print("=" * 75)
    
    df_promise = load_promise_exp()
    texts = df_promise['text'].tolist()
    labels = df_promise['label'].tolist()
    unique_labels = sorted(list(set(labels)))
    
    methods = ['raw', 'barak_or', 'ali_et_al', 'proposed']
    
    tasks = []
    for repeat in range(n_repeats):
        seed = 2000 + repeat * 37
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold, (train_idx, test_idx) in enumerate(skf.split(texts, labels)):
            fold_tr_t = [texts[i] for i in train_idx]
            fold_tr_l = [labels[i] for i in train_idx]
            fold_te_t = [texts[i] for i in test_idx]
            fold_te_l = [labels[i] for i in test_idx]
            for m in methods:
                tasks.append((repeat, fold, m, fold_tr_t, fold_tr_l, fold_te_t, fold_te_l, seed + fold))
                
    print(f"Total parallel tasks: {len(tasks)} (each evaluating ALL 6 classifiers).")
    
    t0 = time.time()
    outputs = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(process_single_fold_all_classifiers)(
            task[2], task[3], task[4], task[5], task[6], unique_labels, task[7]
        ) for task in tasks
    )
    t_elapsed = time.time() - t0
    print(f"\nCompleted all {len(tasks)} multi-classifier tasks in {t_elapsed:.2f} seconds!")
    
    # Restructure results: results[clf][method][metric]
    master_results = {clf: {m: {'accuracy': [], 'f1_score': [], 'precision': [], 'recall': [], 'specificity': [], 'mcc': []} for m in methods} for clf in CLASSIFIER_NAMES}
    
    for task, res_dict in zip(tasks, outputs):
        m = task[2]
        for clf in CLASSIFIER_NAMES:
            for metric in ['accuracy', 'f1_score', 'precision', 'recall', 'specificity', 'mcc']:
                master_results[clf][m][metric].append(res_dict[clf][metric])
                
    # Aggregate stats and compute t-test, Cohen's d, ANOVA
    final_report = {
        'metadata': {'n_repeats': n_repeats, 'n_splits': n_splits, 'elapsed': t_elapsed},
        'classifiers': {},
        'generalization': {}
    }
    
    print("\n" + "=" * 80)
    print(f"{'CLASSIFIER':<20} | {'METHOD':<12} | {'ACCURACY (Mean ± SD)':<22} | {'F1-SCORE':<10} | {'MCC':<8}")
    print("=" * 80)
    
    for clf in CLASSIFIER_NAMES:
        final_report['classifiers'][clf] = {'metrics': {}, 'statistical_tests': {}}
        for m in methods:
            final_report['classifiers'][clf]['metrics'][m] = {}
            for metric in ['accuracy', 'f1_score', 'precision', 'recall', 'specificity', 'mcc']:
                vals = master_results[clf][m][metric]
                # Compute repeat means (30 means)
                rep_means = [float(np.mean(vals[r * n_splits : (r + 1) * n_splits])) for r in range(n_repeats)]
                final_report['classifiers'][clf]['metrics'][m][metric] = {
                    'mean': float(np.mean(rep_means)),
                    'std': float(np.std(rep_means)),
                    'repeat_means': rep_means
                }
            acc_m = final_report['classifiers'][clf]['metrics'][m]['accuracy']['mean'] * 100
            acc_s = final_report['classifiers'][clf]['metrics'][m]['accuracy']['std'] * 100
            f1_m = final_report['classifiers'][clf]['metrics'][m]['f1_score']['mean'] * 100
            mcc_m = final_report['classifiers'][clf]['metrics'][m]['mcc']['mean']
            print(f"{clf:<20} | {m:<12} | {acc_m:.2f}% ± {acc_s:.2f}%          | {f1_m:.2f}%     | {mcc_m:.4f}")
            
        # Statistical tests (Proposed vs. others across the 30 repeated splits)
        prop_accs = final_report['classifiers'][clf]['metrics']['proposed']['accuracy']['repeat_means']
        for base in ['raw', 'barak_or', 'ali_et_al']:
            b_accs = final_report['classifiers'][clf]['metrics'][base]['accuracy']['repeat_means']
            final_report['classifiers'][clf]['statistical_tests'][f'proposed_vs_{base}'] = compute_statistical_tests(prop_accs, b_accs)
            
    # Zero-shot Generalization on FNFC and nfr
    print("\n" + "=" * 80)
    print("EVALUATING GENERALIZATION ON FNFC (7,034 samples) & NFR (625 samples)")
    print("=" * 80)
    df_fnfc = load_fnfc()
    df_nfr = load_nfr()
    
    train_texts = df_promise['text'].tolist()
    train_labels = df_promise['label'].tolist()
    
    final_report['generalization'] = {'FNFC': {}, 'nfr': {}}
    for m in methods:
        res_fnfc = process_single_fold_all_classifiers(
            m, train_texts, train_labels,
            df_fnfc['text'].tolist(), df_fnfc['label'].tolist(),
            unique_labels, 42
        )
        final_report['generalization']['FNFC'][m] = res_fnfc
        
        res_nfr = process_single_fold_all_classifiers(
            m, train_texts, train_labels,
            df_nfr['text'].tolist(), df_nfr['label'].tolist(),
            unique_labels, 42
        )
        final_report['generalization']['nfr'][m] = res_nfr
        
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'all_classifiers_30_split_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, indent=2)
    print(f"\nSaved all results to: {out_path}")
    return final_report


if __name__ == '__main__':
    run_full_suite(n_repeats=30, n_splits=10, n_jobs=12)
