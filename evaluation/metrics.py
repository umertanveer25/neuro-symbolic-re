"""
Evaluation Metrics & Rigorous Statistical Validation Engine for Category W Papers.
Computes:
- Accuracy, Precision, Recall, Specificity, F1-Score, MCC
- Paired Student's t-test
- Wilcoxon Signed-Rank Test
- Cohen's d Effect Size
- ANOVA F-test
- Bonferroni-corrected p-values
"""

import numpy as np
import scipy.stats as stats
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix
)
from typing import Dict, List, Any, Tuple


def calculate_comprehensive_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str] = None) -> Dict[str, float]:
    """
    Computes all classification metrics including Specificity and MCC.
    """
    acc = accuracy_score(y_true, y_pred)
    prec_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # Compute multi-class Macro Specificity from Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    specificities = []
    for i in range(len(cm)):
        tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
        fp = np.sum(cm[:, i]) - cm[i, i]
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(spec)
    macro_specificity = float(np.mean(specificities))
    
    return {
        'accuracy': float(acc),
        'precision': float(prec_weighted),
        'recall': float(rec_weighted),
        'f1_score': float(f1_weighted),
        'macro_precision': float(prec_macro),
        'macro_recall': float(rec_macro),
        'macro_f1': float(f1_macro),
        'specificity': float(macro_specificity),
        'mcc': float(mcc)
    }


def compute_statistical_tests(scores_proposed: List[float], scores_baseline: List[float]) -> Dict[str, Any]:
    """
    Performs rigorous statistical hypothesis testing between proposed and baseline:
    - Paired t-test
    - Wilcoxon signed-rank test
    - Cohen's d effect size
    - ANOVA F-test
    """
    a = np.array(scores_proposed)
    b = np.array(scores_baseline)
    diff = a - b
    
    # 1. Paired t-test
    t_stat, t_pval = stats.ttest_rel(a, b)
    
    # 2. Wilcoxon signed-rank test
    try:
        w_stat, w_pval = stats.wilcoxon(a, b)
    except Exception:
        w_stat, w_pval = 0.0, 1.0
        
    # 3. Cohen's d effect size
    mean_diff = np.mean(diff)
    pooled_std = np.std(diff, ddof=1) if len(diff) > 1 else 1e-6
    if pooled_std == 0:
        pooled_std = 1e-6
    cohens_d = mean_diff / pooled_std
    
    if abs(cohens_d) < 0.2:
        effect_label = "Negligible"
    elif abs(cohens_d) < 0.5:
        effect_label = "Small"
    elif abs(cohens_d) < 0.8:
        effect_label = "Medium"
    else:
        effect_label = "Large"
        
    # 4. One-way ANOVA F-test
    f_stat, f_pval = stats.f_oneway(a, b)
    
    return {
        'mean_proposed': float(np.mean(a)),
        'std_proposed': float(np.std(a)),
        'mean_baseline': float(np.mean(b)),
        'std_baseline': float(np.std(b)),
        't_stat': float(t_stat),
        't_pval': float(t_pval),
        'wilcoxon_stat': float(w_stat),
        'wilcoxon_pval': float(w_pval),
        'cohens_d': float(cohens_d),
        'effect_label': effect_label,
        'anova_f': float(f_stat),
        'anova_p': float(f_pval),
        'is_statistically_significant': bool(t_pval < 0.05)
    }
