"""
Comprehensive Benchmark with Multi-k KNN (k=3, k=5, k=7) and all base paper algorithms:
1. Logistic Regression
2. Linear SVM
3. Random Forest
4. Decision Tree
5. Multinomial Naive Bayes
6. KNN (k=3)
7. KNN (k=5)
8. KNN (k=7)

Evaluated across:
- Raw Baseline
- Barak Or (2025) [SMOTE-Tomek]
- Ali et al. (2026) [Hybrid TF-IDF + SBERT/SVD with SMOTE-Tomek]
- Proposed [Neuro-Symbolic ISO 25010 Synergy]

10-Fold Stratified Cross-Validation in parallel across 12 CPU cores.
"""

import os
import sys
import numpy as np
import pandas as pd
from collections import Counter
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.dataset_loader import load_promise_exp
from neuro.generator import EnhancedNeuroSymbolicAugmenter
from evaluation.metrics import calculate_comprehensive_metrics

CLASSIFIER_NAMES = [
    'logistic_regression',
    'linear_svm',
    'random_forest',
    'decision_tree',
    'naive_bayes',
    'knn_k3',
    'knn_k5',
    'knn_k7'
]


def evaluate_all_models(X_tr, y_tr, X_te, y_te, unique_labels, seed):
    out = {}
    # 1. Logistic Regression
    lr = LogisticRegression(max_iter=600, random_state=seed, C=1.5)
    lr.fit(X_tr, y_tr)
    out['logistic_regression'] = calculate_comprehensive_metrics(y_te, lr.predict(X_te), unique_labels)
    
    # 2. Linear SVM
    svm = LinearSVC(random_state=seed, C=1.0, max_iter=1500)
    svm.fit(X_tr, y_tr)
    out['linear_svm'] = calculate_comprehensive_metrics(y_te, svm.predict(X_te), unique_labels)
    
    # 3. Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1)
    rf.fit(X_tr, y_tr)
    out['random_forest'] = calculate_comprehensive_metrics(y_te, rf.predict(X_te), unique_labels)
    
    # 4. Decision Tree
    dt = DecisionTreeClassifier(random_state=seed, max_depth=25)
    dt.fit(X_tr, y_tr)
    out['decision_tree'] = calculate_comprehensive_metrics(y_te, dt.predict(X_te), unique_labels)
    
    # 5. Naive Bayes
    scaler = MinMaxScaler()
    X_tr_pos = scaler.fit_transform(X_tr)
    X_te_pos = scaler.transform(X_te)
    nb = MultinomialNB(alpha=0.5)
    nb.fit(X_tr_pos, y_tr)
    out['naive_bayes'] = calculate_comprehensive_metrics(y_te, nb.predict(X_te_pos), unique_labels)
    
    # 6. KNN (k=3)
    knn3 = KNeighborsClassifier(n_neighbors=3, n_jobs=1)
    knn3.fit(X_tr, y_tr)
    out['knn_k3'] = calculate_comprehensive_metrics(y_te, knn3.predict(X_te), unique_labels)
    
    # 7. KNN (k=5)
    knn5 = KNeighborsClassifier(n_neighbors=5, n_jobs=1)
    knn5.fit(X_tr, y_tr)
    out['knn_k5'] = calculate_comprehensive_metrics(y_te, knn5.predict(X_te), unique_labels)
    
    # 8. KNN (k=7)
    knn7 = KNeighborsClassifier(n_neighbors=7, n_jobs=1)
    knn7.fit(X_tr, y_tr)
    out['knn_k7'] = calculate_comprehensive_metrics(y_te, knn7.predict(X_te), unique_labels)
    
    return out


def run_pipeline_fold(method, tr_texts, tr_labels, te_texts, te_labels, unique_labels, seed):
    if method == 'raw':
        tfidf = TfidfVectorizer(max_features=1200, ngram_range=(1, 2), stop_words='english')
        X_tr = tfidf.fit_transform(tr_texts).toarray()
        y_tr = np.array(tr_labels)
        X_te = tfidf.transform(te_texts).toarray()
        y_te = np.array(te_labels)
        
    elif method == 'barak_or':
        tfidf = TfidfVectorizer(max_features=1200, stop_words='english')
        X_raw = tfidf.fit_transform(tr_texts).toarray()
        y_raw = np.array(tr_labels)
        min_c = min(Counter(y_raw).values())
        k_neighbors = max(1, min(3, min_c - 1))
        smt = SMOTETomek(smote=SMOTE(k_neighbors=k_neighbors, random_state=seed), tomek=TomekLinks(), random_state=seed)
        try:
            X_tr, y_tr = smt.fit_resample(X_raw, y_raw)
        except Exception:
            X_tr, y_tr = X_raw, y_raw
        X_te = tfidf.transform(te_texts).toarray()
        y_te = np.array(te_labels)
        
    elif method == 'ali_et_al':
        tfidf = TfidfVectorizer(max_features=1200, ngram_range=(1, 2), stop_words='english')
        svd = TruncatedSVD(n_components=40, random_state=seed)
        tf_tr = tfidf.fit_transform(tr_texts)
        svd_tr = svd.fit_transform(tf_tr)
        X_raw = np.hstack([tf_tr.toarray(), svd_tr])
        y_raw = np.array(tr_labels)
        min_c = min(Counter(y_raw).values())
        k_neighbors = max(1, min(3, min_c - 1))
        smt = SMOTETomek(smote=SMOTE(k_neighbors=k_neighbors, random_state=seed), tomek=TomekLinks(), random_state=seed)
        try:
            X_tr, y_tr = smt.fit_resample(X_raw, y_raw)
        except Exception:
            X_tr, y_tr = X_raw, y_raw
        tf_te = tfidf.transform(te_texts)
        svd_te = svd.transform(tf_te)
        X_te = np.hstack([tf_te.toarray(), svd_te])
        y_te = np.array(te_labels)
        
    elif method == 'proposed':
        augmenter = EnhancedNeuroSymbolicAugmenter(random_seed=seed)
        counts = Counter(tr_labels)
        synth_t, synth_l = [], []
        for cls, cnt in counts.items():
            if cls == 'F':
                continue
            if cnt < 40:
                gen_texts, _ = augmenter.generate_verified_batch(cls, 25)
                synth_t.extend(gen_texts)
                synth_l.extend([cls] * len(gen_texts))
                
        combined_tr_t = tr_texts + synth_t
        combined_tr_l = tr_labels + synth_l
        
        tfidf = TfidfVectorizer(max_features=1200, ngram_range=(1, 2), sublinear_tf=True, stop_words='english')
        X_raw = tfidf.fit_transform(combined_tr_t).toarray()
        y_raw = np.array(combined_tr_l)
        min_c = min(Counter(y_raw).values())
        k_neighbors = max(1, min(3, min_c - 1))
        smt = SMOTETomek(smote=SMOTE(k_neighbors=k_neighbors, random_state=seed), tomek=TomekLinks(), random_state=seed)
        try:
            X_tr, y_tr = smt.fit_resample(X_raw, y_raw)
        except Exception:
            X_tr, y_tr = X_raw, y_raw
        X_te = tfidf.transform(te_texts).toarray()
        y_te = np.array(te_labels)
    else:
        raise ValueError(f"Unknown method {method}")
        
    return evaluate_all_models(X_tr, y_tr, X_te, y_te, unique_labels, seed)


def run_benchmark():
    print("=" * 80)
    print("RUNNING BENCHMARK WITH MULTI-k KNN (k=3, k=5, k=7) ACROSS ALL PAPERS")
    print("Pipelines: Raw Baseline | Barak Or (2025) | Ali et al. (2026) | Proposed Synergy")
    print("=" * 80)
    
    df = load_promise_exp()
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    unique_labels = sorted(list(set(labels)))
    
    methods = ['raw', 'barak_or', 'ali_et_al', 'proposed']
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    tasks = []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(texts, labels)):
        tr_t = [texts[i] for i in tr_idx]
        tr_l = [labels[i] for i in tr_idx]
        te_t = [texts[i] for i in te_idx]
        te_l = [labels[i] for i in te_idx]
        for m in methods:
            tasks.append((m, tr_t, tr_l, te_t, te_l, fold + 300))
            
    print(f"Executing {len(tasks)} parallel fold tasks on 12 CPU cores...")
    outputs = Parallel(n_jobs=12, verbose=5)(
        delayed(run_pipeline_fold)(
            task[0], task[1], task[2], task[3], task[4], unique_labels, task[5]
        ) for task in tasks
    )
    
    # Restructure: res[clf][method]['acc']
    res = {clf: {m: {'acc': [], 'f1': [], 'mcc': []} for m in methods} for clf in CLASSIFIER_NAMES}
    for task, fold_out in zip(tasks, outputs):
        m = task[0]
        for clf in CLASSIFIER_NAMES:
            res[clf][m]['acc'].append(fold_out[clf]['accuracy'])
            res[clf][m]['f1'].append(fold_out[clf]['f1_score'])
            res[clf][m]['mcc'].append(fold_out[clf]['mcc'])
            
    print("\n" + "=" * 90)
    print(f"{'CLASSIFIER':<22} | {'RAW BASE':<12} | {'BARAK OR':<12} | {'ALI ET AL':<12} | {'PROPOSED':<14} | {'BEST'}")
    print("=" * 90)
    
    for clf in CLASSIFIER_NAMES:
        raw_a = np.mean(res[clf]['raw']['acc']) * 100
        barak_a = np.mean(res[clf]['barak_or']['acc']) * 100
        ali_a = np.mean(res[clf]['ali_et_al']['acc']) * 100
        prop_a = np.mean(res[clf]['proposed']['acc']) * 100
        
        scores = {'Raw': raw_a, 'Barak Or': barak_a, 'Ali et al': ali_a, 'Proposed': prop_a}
        best_name = max(scores, key=scores.get)
        
        winner_mark = "[WINNER]" if best_name == 'Proposed' else ""
        print(f"{clf:<22} | {raw_a:.2f}%     | {barak_a:.2f}%     | {ali_a:.2f}%     | {prop_a:.2f}%        | {best_name} {winner_mark}")


if __name__ == '__main__':
    run_benchmark()
