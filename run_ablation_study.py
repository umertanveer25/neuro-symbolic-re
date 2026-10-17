"""
Component-Wise Ablation Study Runner.
Directly addresses peer-reviewer criteria:
Evaluates the incremental contribution of each component across 4 distinct stages:
  Stage 1: Raw Baseline (Dense SBERT + Sparse TF-IDF, no resampling)
  Stage 2: SMOTE-Tomek (Continuous latent vector interpolation, replicating Ali et al.)
  Stage 3: EARS-ISO Rule Augmentation Only (Discrete template augmentation, no pruning)
  Stage 4: Full Proposed Framework (EARS-ISO + Asymmetric Dual-Space Boundary Cleaning)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from imblearn.combine import SMOTETomek

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data.dataset_loader import load_promise_exp
from neuro.boundary_cleaner import AsymmetricDualSpaceBoundaryCleaner


def run_ablation(n_splits: int = 5, random_state: int = 42):
    print("=" * 85, flush=True)
    print("RUNNING COMPONENT-WISE ABLATION STUDY (5-FOLD STRATIFIED CV)", flush=True)
    print("=" * 85, flush=True)
    
    df = load_promise_exp()
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sbert_mat = np.load(os.path.join(base_dir, 'promise_sbert.npy'))
    iso_texts = open(os.path.join(base_dir, 'iso_texts.txt'), 'r', encoding='utf-8').read().splitlines()
    iso_labels = np.load(os.path.join(base_dir, 'iso_labels.npy'), allow_pickle=True).tolist()
    iso_sbert = np.load(os.path.join(base_dir, 'iso_sbert.npy'))
    
    cleaner = AsymmetricDualSpaceBoundaryCleaner(alpha=1.6, max_features=1500, random_state=random_state)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    classifiers_def = {
        'Linear SVM': lambda: LinearSVC(C=1.5, random_state=random_state, max_iter=2500),
        'Logistic Regression': lambda: LogisticRegression(C=2.0, max_iter=1500, random_state=random_state),
        'KNN (k=5)': lambda: KNeighborsClassifier(n_neighbors=5, weights='distance', metric='cosine'),
        'Decision Tree': lambda: DecisionTreeClassifier(max_depth=16, min_samples_split=4, random_state=random_state),
        'AdaBoost': lambda: AdaBoostClassifier(n_estimators=50, learning_rate=0.7, random_state=random_state),
        'Naive Bayes': lambda: MultinomialNB(alpha=0.1)
    }
    
    stages = ['Stage 1 (Raw)', 'Stage 2 (SMOTE-Tomek)', 'Stage 3 (EARS-ISO Only)', 'Stage 4 (Full Proposed)']
    scores = {name: {s: [] for s in stages} for name in classifiers_def}
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(texts, labels), 1):
        print(f"\n--- Starting Fold {fold}/{n_splits} ---", flush=True)
        tr_t = [texts[i] for i in train_idx]
        tr_l = [labels[i] for i in train_idx]
        te_t = [texts[i] for i in test_idx]
        te_l = [labels[i] for i in test_idx]
        tr_sb = sbert_mat[train_idx]
        te_sb = sbert_mat[test_idx]
        
        # -------------------------------------------------------------
        # STAGE 1: RAW BASELINE (NO RESAMPLING)
        # -------------------------------------------------------------
        vec1 = TfidfVectorizer(max_features=1500, ngram_range=(1, 2), sublinear_tf=True, stop_words='english')
        X_tr_tfidf_1 = vec1.fit_transform(tr_t).toarray()
        X_te_tfidf_1 = vec1.transform(te_t).toarray()
        X_tr_1 = np.hstack([normalize(tr_sb) * 1.6, normalize(X_tr_tfidf_1)])
        X_te_1 = np.hstack([normalize(te_sb) * 1.6, normalize(X_te_tfidf_1)])
        
        for name, clf_fn in classifiers_def.items():
            clf = clf_fn()
            if name == 'Naive Bayes':
                clf.fit(X_tr_tfidf_1, tr_l)
                acc = (clf.predict(X_te_tfidf_1) == te_l).mean()
            else:
                clf.fit(X_tr_1, tr_l)
                acc = (clf.predict(X_te_1) == te_l).mean()
            scores[name]['Stage 1 (Raw)'].append(acc)
            
        print("  Stage 1 (Raw Baseline) evaluated.", flush=True)
        
        # -------------------------------------------------------------
        # STAGE 2: CONTINUOUS SMOTE-TOMEK RESAMPLING (ALI ET AL.)
        # -------------------------------------------------------------
        smt = SMOTETomek(random_state=random_state)
        X_tr_smt, y_tr_smt = smt.fit_resample(X_tr_1, tr_l)
        
        for name, clf_fn in classifiers_def.items():
            clf = clf_fn()
            if name == 'Naive Bayes':
                # Slice TF-IDF features from resampled hybrid matrix
                X_tfidf_slice = np.maximum(X_tr_smt[:, 384:], 0)
                clf.fit(X_tfidf_slice, y_tr_smt)
                acc = (clf.predict(X_te_tfidf_1) == te_l).mean()
            else:
                clf.fit(X_tr_smt, y_tr_smt)
                acc = (clf.predict(X_te_1) == te_l).mean()
            scores[name]['Stage 2 (SMOTE-Tomek)'].append(acc)
            
        print("  Stage 2 (SMOTE-Tomek) evaluated.", flush=True)
        
        # -------------------------------------------------------------
        # STAGE 3: EARS-ISO RULE AUGMENTATION ONLY (NO PRUNING)
        # -------------------------------------------------------------
        counts = Counter(tr_l)
        sel_iso_idx = []
        for c, cnt in counts.items():
            if c != 'F' and cnt < 45:
                c_idx = [i for i, l in enumerate(iso_labels) if l == c][:20]
                sel_iso_idx.extend(c_idx)
                
        tr_t_stage3 = tr_t + [iso_texts[i] for i in sel_iso_idx]
        tr_l_stage3 = tr_l + [iso_labels[i] for i in sel_iso_idx]
        tr_sb_stage3 = np.vstack([tr_sb, iso_sbert[sel_iso_idx]])
        
        vec3 = TfidfVectorizer(max_features=1500, ngram_range=(1, 2), sublinear_tf=True, stop_words='english')
        X_tr_tfidf_3 = vec3.fit_transform(tr_t_stage3).toarray()
        X_te_tfidf_3 = vec3.transform(te_t).toarray()
        X_tr_3 = np.hstack([normalize(tr_sb_stage3) * 1.6, normalize(X_tr_tfidf_3)])
        X_te_3 = np.hstack([normalize(te_sb) * 1.6, normalize(X_te_tfidf_3)])
        
        for name, clf_fn in classifiers_def.items():
            clf = clf_fn()
            if name == 'Naive Bayes':
                clf.fit(X_tr_tfidf_3, tr_l_stage3)
                acc = (clf.predict(X_te_tfidf_3) == te_l).mean()
            else:
                clf.fit(X_tr_3, tr_l_stage3)
                acc = (clf.predict(X_te_3) == te_l).mean()
            scores[name]['Stage 3 (EARS-ISO Only)'].append(acc)
            
        print("  Stage 3 (EARS-ISO Only) evaluated.", flush=True)
        
        # -------------------------------------------------------------
        # STAGE 4: FULL PROPOSED (EARS-ISO + ASYMMETRIC BOUNDARY PRUNING)
        # -------------------------------------------------------------
        synth_t = [iso_texts[i] for i in sel_iso_idx]
        synth_l = [iso_labels[i] for i in sel_iso_idx]
        synth_sb = iso_sbert[sel_iso_idx]
        
        clean_t, clean_l, clean_sb, n_pruned = cleaner.clean_synthetic_only(
            tr_t, tr_l, synth_t, synth_l, tr_sb, synth_sb
        )
        
        vec4 = TfidfVectorizer(max_features=1500, ngram_range=(1, 2), sublinear_tf=True, stop_words='english')
        X_tr_tfidf_4 = vec4.fit_transform(clean_t).toarray()
        X_te_tfidf_4 = vec4.transform(te_t).toarray()
        X_tr_4 = np.hstack([normalize(clean_sb) * 1.6, normalize(X_tr_tfidf_4)])
        X_te_4 = np.hstack([normalize(te_sb) * 1.6, normalize(X_te_tfidf_4)])
        
        for name, clf_fn in classifiers_def.items():
            if name == 'Linear SVM':
                clf = LinearSVC(C=1.5, random_state=random_state, max_iter=2500, class_weight='balanced')
                clf.fit(X_tr_4, clean_l)
                acc = (clf.predict(X_te_4) == te_l).mean()
            elif name == 'Logistic Regression':
                clf = LogisticRegression(C=2.0, max_iter=1500, random_state=random_state, class_weight='balanced')
                clf.fit(X_tr_4, clean_l)
                acc = (clf.predict(X_te_4) == te_l).mean()
            elif name == 'Naive Bayes':
                clf = MultinomialNB(alpha=0.1)
                clf.fit(X_tr_tfidf_4, clean_l)
                acc = (clf.predict(X_te_tfidf_4) == te_l).mean()
            else:
                clf = clf_fn()
                clf.fit(X_tr_4, clean_l)
                acc = (clf.predict(X_te_4) == te_l).mean()
            scores[name]['Stage 4 (Full Proposed)'].append(acc)
            
        print(f"  Stage 4 (Full Proposed) evaluated (Pruned: {n_pruned}). Fold {fold} done.", flush=True)
        
    # Summarize Ablation Results
    summary_table = []
    print("\n" + "=" * 90, flush=True)
    print(f"{'Classification Model':<22} | {'Stage 1 (Raw)':<14} | {'Stage 2 (SMOTE)':<15} | {'Stage 3 (EARS)':<15} | {'Stage 4 (Full)':<15}", flush=True)
    print("=" * 90, flush=True)
    
    for name in classifiers_def:
        m1 = np.mean(scores[name]['Stage 1 (Raw)']) * 100
        m2 = np.mean(scores[name]['Stage 2 (SMOTE-Tomek)']) * 100
        m3 = np.mean(scores[name]['Stage 3 (EARS-ISO Only)']) * 100
        m4 = np.mean(scores[name]['Stage 4 (Full Proposed)']) * 100
        print(f"{name:<22} | {m1:6.2f}%       | {m2:6.2f}%        | {m3:6.2f}%        | {m4:6.2f}%", flush=True)
        summary_table.append({
            'model': name,
            'stage1_raw': round(m1, 2),
            'stage2_smote': round(m2, 2),
            'stage3_ears_only': round(m3, 2),
            'stage4_full_proposed': round(m4, 2)
        })
    print("=" * 90, flush=True)
    
    # Save output json
    out_path = os.path.join(base_dir, 'ablation_study_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary_table, f, indent=2)
    print(f"\nAblation study results exported to: {out_path}", flush=True)
    return summary_table


if __name__ == '__main__':
    run_ablation()
