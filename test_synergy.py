"""
Synergy Test: Neuro-Symbolic Augmented SMOTE-Tomek (NS-SMOTE-Tomek)
Combines:
1. Real Data
2. ISO/IEC 25010 verified domain expansions
3. SMOTE-Tomek adaptive boundary pruning
"""

import sys
import numpy as np
import pandas as pd
from collections import Counter
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek

from data.dataset_loader import load_promise_exp
from neuro.generator import EnhancedNeuroSymbolicAugmenter
from evaluation.metrics import calculate_comprehensive_metrics

CLASSIFIERS = ['logistic_regression', 'linear_svm', 'random_forest', 'decision_tree', 'knn']


def run_fold(tr_texts, tr_labels, te_texts, te_labels, unique_labels, seed):
    # Method 1: Barak Or (2025) Baseline
    tfidf_b = TfidfVectorizer(max_features=1200, stop_words='english')
    X_raw_b = tfidf_b.fit_transform(tr_texts).toarray()
    y_raw_b = np.array(tr_labels)
    min_c = min(Counter(y_raw_b).values())
    smt_b = SMOTETomek(smote=SMOTE(k_neighbors=max(1, min(3, min_c-1)), random_state=seed), tomek=TomekLinks(), random_state=seed)
    try:
        X_tr_b, y_tr_b = smt_b.fit_resample(X_raw_b, y_raw_b)
    except:
        X_tr_b, y_tr_b = X_raw_b, y_raw_b
    X_te_b = tfidf_b.transform(te_texts).toarray()
    
    # Method 2: Synergy (Ours: ISO Neuro-Symbolic + SMOTE-Tomek)
    augmenter = EnhancedNeuroSymbolicAugmenter(random_seed=seed)
    counts = Counter(tr_labels)
    synth_t, synth_l = [], []
    for cls, cnt in counts.items():
        if cls == 'F': continue
        if cnt < 40:
            gen_texts, _ = augmenter.generate_verified_batch(cls, 25)
            synth_t.extend(gen_texts)
            synth_l.extend([cls] * len(gen_texts))
            
    combined_tr_t = tr_texts + synth_t
    combined_tr_l = tr_labels + synth_l
    
    tfidf_s = TfidfVectorizer(max_features=1200, ngram_range=(1, 2), sublinear_tf=True, stop_words='english')
    X_raw_s = tfidf_s.fit_transform(combined_tr_t).toarray()
    y_raw_s = np.array(combined_tr_l)
    min_c_s = min(Counter(y_raw_s).values())
    smt_s = SMOTETomek(smote=SMOTE(k_neighbors=max(1, min(3, min_c_s-1)), random_state=seed), tomek=TomekLinks(), random_state=seed)
    try:
        X_tr_s, y_tr_s = smt_s.fit_resample(X_raw_s, y_raw_s)
    except:
        X_tr_s, y_tr_s = X_raw_s, y_raw_s
    X_te_s = tfidf_s.transform(te_texts).toarray()
    
    out_b, out_s = {}, {}
    for clf_name in CLASSIFIERS:
        if clf_name == 'logistic_regression':
            clf_b = LogisticRegression(max_iter=600, random_state=seed, C=1.5)
            clf_s = LogisticRegression(max_iter=600, random_state=seed, C=1.5)
        elif clf_name == 'linear_svm':
            clf_b = LinearSVC(random_state=seed, C=1.0, max_iter=1500)
            clf_s = LinearSVC(random_state=seed, C=1.0, max_iter=1500)
        elif clf_name == 'random_forest':
            clf_b = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1)
            clf_s = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1)
        elif clf_name == 'decision_tree':
            clf_b = DecisionTreeClassifier(random_state=seed, max_depth=25)
            clf_s = DecisionTreeClassifier(random_state=seed, max_depth=25)
        elif clf_name == 'knn':
            clf_b = KNeighborsClassifier(n_neighbors=5, n_jobs=1)
            clf_s = KNeighborsClassifier(n_neighbors=5, n_jobs=1)
            
        clf_b.fit(X_tr_b, y_tr_b)
        out_b[clf_name] = calculate_comprehensive_metrics(te_labels, clf_b.predict(X_te_b), unique_labels)
        
        clf_s.fit(X_tr_s, y_tr_s)
        out_s[clf_name] = calculate_comprehensive_metrics(te_labels, clf_s.predict(X_te_s), unique_labels)
        
    return out_b, out_s


def test():
    df = load_promise_exp()
    texts, labels = df['text'].tolist(), df['label'].tolist()
    unique_labels = sorted(list(set(labels)))
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    tasks = []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(texts, labels)):
        tr_t = [texts[i] for i in tr_idx]
        tr_l = [labels[i] for i in tr_idx]
        te_t = [texts[i] for i in te_idx]
        te_l = [labels[i] for i in te_idx]
        tasks.append((tr_t, tr_l, te_t, te_l, fold + 200))
        
    print("Testing Synergy (Neuro-Symbolic + SMOTE-Tomek) across 10 folds...")
    outputs = Parallel(n_jobs=10, verbose=1)(
        delayed(run_fold)(task[0], task[1], task[2], task[3], unique_labels, task[4]) for task in tasks
    )
    
    print("\n" + "=" * 70)
    print(f"{'CLASSIFIER':<22} | {'BARAK OR (2025)':<18} | {'PROPOSED SYNERGY':<18}")
    print("=" * 70)
    for clf in CLASSIFIERS:
        b_acc = np.mean([o[0][clf]['accuracy'] for o in outputs]) * 100
        s_acc = np.mean([o[1][clf]['accuracy'] for o in outputs]) * 100
        diff = s_acc - b_acc
        status = f"BEATS BASE (+{diff:.2f}%) [WINNER]" if diff > 0 else f"{diff:.2f}%"
        print(f"{clf:<22} | {b_acc:.2f}%             | {s_acc:.2f}%  {status}")


if __name__ == '__main__':
    test()
