"""
Test Verification of Improved Neuro-Symbolic Pipeline:
- Full Proportional Balancing (Matching majority class ratios)
- High-Entropy Lexical Slot Expansion
- Asymmetric Boundary Pruning (100% Real Samples Protected)
Runs 10-fold cross-validation across all 6 classifiers in parallel.
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

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.dataset_loader import load_promise_exp
from neuro.generator import EnhancedNeuroSymbolicAugmenter
from neuro.boundary_cleaner import AsymmetricDualSpaceBoundaryCleaner
from evaluation.metrics import calculate_comprehensive_metrics

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek

CLASSIFIERS = ['logistic_regression', 'linear_svm', 'random_forest', 'naive_bayes', 'decision_tree', 'knn']


def evaluate_classifiers(X_train, y_train, X_test, y_test, unique_labels, seed):
    out = {}
    # LR
    lr = LogisticRegression(max_iter=600, random_state=seed, C=2.0)
    lr.fit(X_train, y_train)
    out['logistic_regression'] = calculate_comprehensive_metrics(y_test, lr.predict(X_test), unique_labels)
    
    # Linear SVM
    svm = LinearSVC(random_state=seed, C=1.0, max_iter=1500)
    svm.fit(X_train, y_train)
    out['linear_svm'] = calculate_comprehensive_metrics(y_test, svm.predict(X_test), unique_labels)
    
    # RF
    rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1)
    rf.fit(X_train, y_train)
    out['random_forest'] = calculate_comprehensive_metrics(y_test, rf.predict(X_test), unique_labels)
    
    # NB
    scaler = MinMaxScaler()
    X_tr_pos = scaler.fit_transform(X_train)
    X_te_pos = scaler.transform(X_test)
    nb = MultinomialNB(alpha=0.5)
    nb.fit(X_tr_pos, y_train)
    out['naive_bayes'] = calculate_comprehensive_metrics(y_test, nb.predict(X_te_pos), unique_labels)
    
    # DT
    dt = DecisionTreeClassifier(random_state=seed, max_depth=25)
    dt.fit(X_train, y_train)
    out['decision_tree'] = calculate_comprehensive_metrics(y_test, dt.predict(X_test), unique_labels)
    
    # KNN
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=1)
    knn.fit(X_train, y_train)
    out['knn'] = calculate_comprehensive_metrics(y_test, knn.predict(X_test), unique_labels)
    
    return out


def run_single_improved_fold(method, tr_texts, tr_labels, te_texts, te_labels, unique_labels, seed):
    if method == 'barak_or':
        tfidf = TfidfVectorizer(max_features=1200, stop_words='english')
        X_raw = tfidf.fit_transform(tr_texts).toarray()
        y_raw = np.array(tr_labels)
        
        min_c = min(Counter(y_raw).values())
        k_neighbors = max(1, min(3, min_c - 1))
        smt = SMOTETomek(smote=SMOTE(k_neighbors=k_neighbors, random_state=seed), tomek=TomekLinks(), random_state=seed)
        try:
            X_train, y_train = smt.fit_resample(X_raw, y_raw)
        except Exception:
            X_train, y_train = X_raw, y_raw
        X_test = tfidf.transform(te_texts).toarray()
        y_test = np.array(te_labels)
        
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
            X_train, y_train = smt.fit_resample(X_raw, y_raw)
        except Exception:
            X_train, y_train = X_raw, y_raw
        tf_te = tfidf.transform(te_texts)
        svd_te = svd.transform(tf_te)
        X_test = np.hstack([tf_te.toarray(), svd_te])
        y_test = np.array(te_labels)
        
    elif method == 'improved_proposed':
        augmenter = EnhancedNeuroSymbolicAugmenter(random_seed=seed)
        cleaner = AsymmetricDualSpaceBoundaryCleaner(random_state=seed)
        
        counts = Counter(tr_labels)
        # Full Proportional Target: balance minority up to majority class scale!
        target_count = int(max(counts.values()) * 0.75)  # e.g., ~280-300 per class
        
        synth_texts = []
        synth_labels = []
        
        for cls, count in counts.items():
            if cls == 'F':
                continue
            if count < target_count:
                needed = target_count - count
                gen_texts, _ = augmenter.generate_verified_batch(cls, needed)
                synth_texts.extend(gen_texts)
                synth_labels.extend([cls] * len(gen_texts))
                
        # Asymmetric cleaning: prune ONLY synthetic artifacts, 100% protect real data!
        final_tr_texts, final_tr_labels, pruned = cleaner.clean_synthetic_only(
            tr_texts, tr_labels, synth_texts, synth_labels
        )
        
        tfidf = TfidfVectorizer(max_features=1500, ngram_range=(1, 2), sublinear_tf=True, stop_words='english')
        svd = TruncatedSVD(n_components=40, random_state=seed)
        
        tf_tr = tfidf.fit_transform(final_tr_texts)
        svd_tr = svd.fit_transform(tf_tr)
        X_train = np.hstack([tf_tr.toarray(), svd_tr])
        y_train = np.array(final_tr_labels)
        
        tf_te = tfidf.transform(te_texts)
        svd_te = svd.transform(tf_te)
        X_test = np.hstack([tf_te.toarray(), svd_te])
        y_test = np.array(te_labels)
        
    return evaluate_classifiers(X_train, y_train, X_test, y_test, unique_labels, seed)


def run_benchmark():
    print("=" * 75)
    print("RUNNING 10-FOLD CV FOR IMPROVED NEURO-SYMBOLIC PIPELINE VS BASE PAPERS")
    print("=" * 75)
    df = load_promise_exp()
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    unique_labels = sorted(list(set(labels)))
    
    methods = ['barak_or', 'ali_et_al', 'improved_proposed']
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    tasks = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(texts, labels)):
        tr_t = [texts[i] for i in train_idx]
        tr_l = [labels[i] for i in train_idx]
        te_t = [texts[i] for i in test_idx]
        te_l = [labels[i] for i in test_idx]
        for m in methods:
            tasks.append((m, tr_t, tr_l, te_t, te_l, fold + 100))
            
    print(f"Executing {len(tasks)} tasks across 12 CPU cores...")
    outputs = Parallel(n_jobs=12, verbose=5)(
        delayed(run_single_improved_fold)(
            task[0], task[1], task[2], task[3], task[4], unique_labels, task[5]
        ) for task in tasks
    )
    
    # Collate
    res = {m: {clf: {'acc': [], 'f1': [], 'mcc': []} for clf in CLASSIFIERS} for m in methods}
    for task, fold_out in zip(tasks, outputs):
        m = task[0]
        for clf in CLASSIFIERS:
            res[m][clf]['acc'].append(fold_out[clf]['accuracy'])
            res[m][clf]['f1'].append(fold_out[clf]['f1_score'])
            res[m][clf]['mcc'].append(fold_out[clf]['mcc'])
            
    print("\n" + "=" * 80)
    print(f"{'CLASSIFIER':<20} | {'BARAK OR (2025)':<18} | {'ALI ET AL (2026)':<18} | {'IMPROVED PROPOSED':<18}")
    print("=" * 80)
    for clf in CLASSIFIERS:
        b_acc = np.mean(res['barak_or'][clf]['acc']) * 100
        a_acc = np.mean(res['ali_et_al'][clf]['acc']) * 100
        p_acc = np.mean(res['improved_proposed'][clf]['acc']) * 100
        
        winner = "PROPOSED 🏆" if p_acc >= max(b_acc, a_acc) else ("Barak Or" if b_acc > a_acc else "Ali et al")
        print(f"{clf:<20} | {b_acc:.2f}%             | {a_acc:.2f}%             | {p_acc:.2f}% ({winner})")


if __name__ == '__main__':
    run_benchmark()
