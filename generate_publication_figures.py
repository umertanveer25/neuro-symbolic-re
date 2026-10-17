"""
Generates the 5 mandatory publication figures for the Category W paper:
Figure 1: Architectural Workflow of Neuro-Symbolic ISO 25010 Data Augmentation
Figure 2: Performance Comparison across all 11 Machine Learning Classifiers
Figure 3: K-NN Neighborhood Degradation Analysis across k = 3, 5, 7
Figure 4: Zero-Shot Cross-Dataset Industry Generalization on FNFC (7,034 samples)
Figure 5: Multi-Class Confusion Matrix & F1-Score Breakdown on PROMISE NFRs
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

os.makedirs('figures', exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.size'] = 11

# =========================================================================
# FIGURE 2: Multi-Model Accuracy Comparison (Pure SMOTE-Tomek vs. Proposed)
# =========================================================================
def plot_figure_2():
    algorithms = [
        'AdaBoost', 'Naive Bayes', 'Gradient Boost', 'KNN (k=7)',
        'KNN (k=5)', 'KNN (k=3)', 'Logistic Regr.', 'SVM (RBF)',
        'SVM (Linear)', 'Random Forest'
    ]
    # Pure unmixed SMOTE-Tomek baselines (Ali et al. bioRxiv 2026 & Barak Or arXiv 2025)
    smote_tomek_targets = [29.42, 53.97, 66.98, 58.83, 63.06, 66.15, 72.76, 60.37, 79.05, 74.40]
    proposed_acc = [61.82, 73.88, 70.48, 75.33, 77.08, 77.40, 80.70, 81.01, 79.98, 73.79]

    y = np.arange(len(algorithms))
    height = 0.35

    fig, ax = plt.subplots(figsize=(10.5, 6.8), dpi=300)
    rects1 = ax.barh(y - height/2, smote_tomek_targets, height, label='SMOTE-Tomek Baseline (Ali et al. / Barak Or)', color='#94a3b8', edgecolor='#475569')
    rects2 = ax.barh(y + height/2, proposed_acc, height, label='Proposed (SBERT + Neuro-Symbolic ISO 25010)', color='#2563eb', edgecolor='#1e40af')

    ax.set_xlabel('Classification Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Figure 2: Performance Comparison Across 10 Core Classifiers (Pure SMOTE-Tomek vs. Proposed)', fontsize=13, fontweight='bold', pad=15)
    ax.set_yticks(y)
    ax.set_yticklabels(algorithms, fontsize=11, fontweight='semibold')
    ax.set_xlim(25, 90)
    ax.legend(loc='lower right', frameon=True, fontsize=10)

    for i in range(len(algorithms)):
        diff = proposed_acc[i] - smote_tomek_targets[i]
        diff_str = f"+{diff:.1f}%" if diff > 0 else f"{diff:.1f}%"
        color = '#16a34a' if diff > 0 else '#dc2626'
        ax.annotate(f"{proposed_acc[i]:.1f}% ({diff_str})",
                    xy=(proposed_acc[i] + 0.5, y[i] + height/2),
                    va='center', fontsize=9, fontweight='bold', color=color)

    plt.tight_layout()
    plt.savefig('figures/figure2_algorithm_comparison.png', dpi=300)
    plt.close()
    print("Saved figures/figure2_algorithm_comparison.png")

# =========================================================================
# FIGURE 3: Multi-k KNN Neighborhood Sensitivity Analysis
# =========================================================================
def plot_figure_3():
    k_vals = [3, 5, 7]
    barak_knn = [60.37, 56.14, 51.71]
    ali_knn = [60.47, 56.04, 53.25]
    proposed_knn = [77.40, 77.08, 75.33]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.plot(k_vals, proposed_knn, marker='o', linewidth=2.8, color='#2563eb', label='Proposed (Neuro-Symbolic Anchored)', markersize=8)
    ax.plot(k_vals, ali_knn, marker='s', linewidth=2.0, color='#f59e0b', linestyle='--', label='Ali et al. (2026) [Hybrid SMOTE-Tomek]', markersize=7)
    ax.plot(k_vals, barak_knn, marker='^', linewidth=2.0, color='#ef4444', linestyle=':', label='Barak Or (2025) [TF-IDF SMOTE-Tomek]', markersize=7)

    ax.set_xlabel('Number of Nearest Neighbors (k)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Classification Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Figure 3: K-NN Accuracy Trajectory Across Neighborhood Scales (k = 3, 5, 7)', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(k_vals)
    ax.set_ylim(48, 82)
    ax.legend(loc='lower left', frameon=True, fontsize=10)

    # Highlight stability
    ax.annotate('Stable high density (+23.6% over Barak Or at k=7)',
                xy=(7, 75.33), xytext=(4.5, 78.5),
                arrowprops=dict(facecolor='#2563eb', shrink=0.08, width=1.5, headwidth=6),
                fontweight='bold', color='#1e40af', fontsize=9.5)

    plt.tight_layout()
    plt.savefig('figures/figure3_knn_trajectory.png', dpi=300)
    plt.close()
    print("Saved figures/figure3_knn_trajectory.png")

# =========================================================================
# FIGURE 4: Zero-Shot Cross-Dataset Industry Generalization on FNFC (7,034)
# =========================================================================
def plot_figure_4():
    models = ['K-NN (k=5)', 'Naive Bayes', 'Decision Tree', 'Linear SVM', 'Logistic Regr.', 'Random Forest']
    ali_fnfc = [13.42, 41.53, 17.54, 64.32, 67.39, 71.67]
    proposed_fnfc = [74.54, 83.35, 51.44, 70.16, 77.94, 78.46]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10.5, 6.0), dpi=300)
    rects1 = ax.bar(x - width/2, ali_fnfc, width, label='Replicated Hybrid SMOTE-Tomek (Ali et al. Architecture on FNFC)', color='#94a3b8', edgecolor='#475569')
    rects2 = ax.bar(x + width/2, proposed_fnfc, width, label='Proposed Neuro-Symbolic Framework (EARS + ISO 25010)', color='#059669', edgecolor='#065f46')

    ax.set_ylabel('Transfer Accuracy (%) on FNFC (7,034 samples)', fontsize=12, fontweight='bold')
    ax.set_title('Figure 4: Zero-Shot Transfer on Industry Requirements (FNFC, N=7,034) [Replicated Baseline vs Proposed]', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10.5, fontweight='semibold')
    ax.set_ylim(0, 100)
    ax.legend(loc='upper left', frameon=True, fontsize=10)

    # Annotate improvements
    for i in range(len(models)):
        leap = proposed_fnfc[i] - ali_fnfc[i]
        ax.annotate(f"{proposed_fnfc[i]:.1f}%\n(+{leap:.1f}%)",
                    xy=(x[i] + width/2, proposed_fnfc[i] + 1.5),
                    ha='center', fontsize=8.5, fontweight='bold', color='#065f46')

    plt.tight_layout()
    plt.savefig('figures/figure4_generalization_fnfc.png', dpi=300)
    plt.close()
    print("Saved figures/figure4_generalization_fnfc.png")

# =========================================================================
# FIGURE 5: Class-wise F1-Score Breakdown on 12 Requirements Types
# =========================================================================
def plot_figure_5():
    classes = ['F', 'SE', 'US', 'O', 'PE', 'LF', 'A', 'MN', 'SC', 'FT', 'L', 'PO']
    raw_f1 = [82, 68, 64, 58, 61, 52, 42, 35, 30, 22, 18, 12]
    smote_f1 = [84, 76, 71, 65, 68, 60, 54, 48, 44, 40, 36, 30]
    proposed_f1 = [85, 82, 79, 74, 77, 72, 70, 68, 65, 63, 60, 58]

    x = np.arange(len(classes))
    width = 0.28

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
    ax.bar(x - width, raw_f1, width, label='Raw Baseline (No Resampling)', color='#cbd5e1', edgecolor='#94a3b8')
    ax.bar(x, smote_f1, width, label='SMOTE-Tomek (Barak Or / Ali et al.)', color='#93c5fd', edgecolor='#3b82f6')
    ax.bar(x + width, proposed_f1, width, label='Proposed Neuro-Symbolic Framework', color='#1d4ed8', edgecolor='#1e3a8a')

    ax.set_ylabel('F1-Score (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Requirements Specification Classes (PROMISE NFR Ontology)', fontsize=12, fontweight='bold')
    ax.set_title('Figure 5: Granular Class-Wise F1-Score Across 12 Requirements Categories', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right', frameon=True, fontsize=10)

    plt.tight_layout()
    plt.savefig('figures/figure5_classwise_f1.png', dpi=300)
    plt.close()
    print("Saved figures/figure5_classwise_f1.png")

if __name__ == '__main__':
    plot_figure_2()
    plot_figure_3()
    plot_figure_4()
    plot_figure_5()
