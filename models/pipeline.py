"""
Experimental Pipeline & Model Benchmarking.
Implements:
1. Raw Baseline (no resampling)
2. Baseline 1: Barak Or (2025) - Standard SMOTE-Tomek
3. Baseline 2: Ali et al. (2026) - Hybrid TF-IDF + Dense SBERT with SMOTE-Tomek
4. Proposed: Explainable Neuro-Symbolic ISO/IEC 25010 + Dual-Space Boundary Filter
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks

from neuro.generator import NeuroSymbolicAugmenter
from neuro.boundary_cleaner import DualSpaceBoundaryCleaner


class HybridFeatureExtractor:
    """
    Extracts hybrid representation:
    - Sparse TF-IDF (n-grams 1-2, sublinear tf)
    - Latent Semantic Analysis (TruncatedSVD)
    """
    def __init__(self, max_features: int = 1500, n_components: int = 50, random_state: int = 42):
        self.tfidf = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), sublinear_tf=True, stop_words='english')
        self.svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        tfidf_mat = self.tfidf.fit_transform(texts)
        svd_mat = self.svd.fit_transform(tfidf_mat)
        # Concatenate sparse TF-IDF with dense semantic projection
        return np.hstack([tfidf_mat.toarray(), svd_mat])
        
    def transform(self, texts: List[str]) -> np.ndarray:
        tfidf_mat = self.tfidf.transform(texts)
        svd_mat = self.svd.transform(tfidf_mat)
        return np.hstack([tfidf_mat.toarray(), svd_mat])


def run_pipeline_experiment(
    train_texts: List[str],
    train_labels: List[str],
    test_texts: List[str],
    test_labels: List[str],
    method: str = 'proposed',
    classifier_name: str = 'logistic_regression',
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Runs a single training and evaluation cycle under one of the 4 methods:
    'raw', 'barak_or', 'ali_et_al', 'proposed'
    """
    aug_stats = {}
    
    # --- Data Augmentation Step (Only applied to training partition to prevent leakage) ---
    if method == 'raw':
        final_train_texts = list(train_texts)
        final_train_labels = list(train_labels)
        
        feat_extractor = HybridFeatureExtractor(random_state=random_state)
        X_train = feat_extractor.fit_transform(final_train_texts)
        y_train = np.array(final_train_labels)
        
    elif method == 'barak_or':
        # Barak Or (2025): standard TF-IDF followed by SMOTE-Tomek
        tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
        X_raw = tfidf.fit_transform(train_texts).toarray()
        y_raw = np.array(train_labels)
        
        # Determine k_neighbors safely based on smallest minority class
        min_class_count = min(Counter(y_raw).values())
        k_neighbors = max(1, min(5, min_class_count - 1))
        
        smote = SMOTE(k_neighbors=k_neighbors, random_state=random_state)
        tomek = TomekLinks()
        smt = SMOTETomek(smote=smote, tomek=tomek, random_state=random_state)
        
        try:
            X_train, y_train = smt.fit_resample(X_raw, y_raw)
        except Exception:
            # Fallback if any class is too tiny
            X_train, y_train = X_raw, y_raw
            
        feat_extractor = tfidf
        
    elif method == 'ali_et_al':
        # Ali et al. (2026): Hybrid TF-IDF + SVD, then SMOTE-Tomek
        feat_extractor = HybridFeatureExtractor(random_state=random_state)
        X_raw = feat_extractor.fit_transform(train_texts)
        y_raw = np.array(train_labels)
        
        min_class_count = min(Counter(y_raw).values())
        k_neighbors = max(1, min(5, min_class_count - 1))
        
        smote = SMOTE(k_neighbors=k_neighbors, random_state=random_state)
        tomek = TomekLinks()
        smt = SMOTETomek(smote=smote, tomek=tomek, random_state=random_state)
        
        try:
            X_train, y_train = smt.fit_resample(X_raw, y_raw)
        except Exception:
            X_train, y_train = X_raw, y_raw
            
    elif method == 'proposed':
        # Proposed: Explainable Neuro-Symbolic ISO 25010 + Dual-Space Boundary Filter
        augmenter = NeuroSymbolicAugmenter(random_seed=random_state)
        cleaner = DualSpaceBoundaryCleaner(random_state=random_state)
        
        counts = Counter(train_labels)
        target_count = max(counts.values()) // 2  # Balance minority up to balanced target
        
        augmented_texts = list(train_texts)
        augmented_labels = list(train_labels)
        
        total_gen_stats = {'attempted': 0, 'passed': 0, 'pruned_tomek': 0}
        
        for cls, count in counts.items():
            if cls == 'F':
                continue  # 'F' is the majority class, does not need synthesis
            if count < target_count:
                needed = min(target_count - count, 40)
                gen_texts, gstats = augmenter.generate_verified_batch(cls, needed)
                augmented_texts.extend(gen_texts)
                augmented_labels.extend([cls] * len(gen_texts))
                total_gen_stats['attempted'] += gstats['total_attempted']
                total_gen_stats['passed'] += gstats['passed_symbolic']
                
        # Dual-space boundary cleaning in latent space
        cleaned_texts, cleaned_labels, pruned = cleaner.fit_and_clean(augmented_texts, augmented_labels)
        total_gen_stats['pruned_tomek'] = pruned
        aug_stats = total_gen_stats
        
        feat_extractor = HybridFeatureExtractor(random_state=random_state)
        X_train = feat_extractor.fit_transform(cleaned_texts)
        y_train = np.array(cleaned_labels)
        
    else:
        raise ValueError(f"Unknown method: {method}")
        
    # --- Feature Transformation on Test Set ---
    if isinstance(feat_extractor, HybridFeatureExtractor):
        X_test = feat_extractor.transform(test_texts)
    else:
        X_test = feat_extractor.transform(test_texts).toarray()
    y_test = np.array(test_labels)
    
    # --- Classifier Selection ---
    if classifier_name == 'logistic_regression':
        clf = LogisticRegression(max_iter=1000, random_state=random_state, C=1.5)
    elif classifier_name == 'random_forest':
        clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    elif classifier_name == 'svm':
        clf = LinearSVC(random_state=random_state, C=1.0, max_iter=2000)
    elif classifier_name == 'gradient_boosting':
        clf = GradientBoostingClassifier(n_estimators=80, random_state=random_state)
    else:
        clf = LogisticRegression(max_iter=1000, random_state=random_state)
        
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    return {
        'model': clf,
        'y_true': y_test,
        'y_pred': y_pred,
        'aug_stats': aug_stats
    }
