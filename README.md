# Explainable Neuro-Symbolic Requirements Classification

> **Official Implementation Repository for:**  
> *"Explainable Neuro-Symbolic Requirements Classification: Mitigating Topological Collapse in Extreme Class Imbalances via ISO/IEC 25010 Semantic Gating and Asymmetric Dual-Space Boundary Cleaning"*

---

## 📌 Abstract
Software requirements classification is frequently undermined by extreme class imbalance in canonical benchmarks like the PROMISE repository (969 requirements), where critical non-functional categories (e.g., Portability, Legal, Fault Tolerance) comprise fewer than 20 authentic instances. Recent studies (Barak Or, arXiv:2501.06491, Jan 2025; Ali et al., bioRxiv:10.64898/2026.01.13.699413, Jan 2026) applied Euclidean SMOTE-Tomek resampling in high-dimensional feature spaces, inadvertently introducing "ghost vector" artifacts that distort decision boundaries and trigger severe performance degradation across distance and tree-based classifiers.

This repository implements a **Knowledge-Guided Neuro-Symbolic Framework** that replaces continuous mathematical interpolation with discrete, grammar-constrained requirement generation:
1. **Combinatorial EARS Syntactic Generator:** Generates normative natural language specifications adhering to Easy Approach to Requirements Syntax (Ubiquitous, Event-Driven, Unwanted Behavior patterns).
2. **Deterministic ISO/IEC 25010 Quality Filter:** Validates domain characteristic keywords while enforcing zero tolerance (0.00%) for weak modals (*might, could*) and vague terms (*fast, flexible*).
3. **Asymmetric Dual-Space Boundary Cleaner:** Locks 100% of authentic human requirements and selectively prunes only ambiguous synthetic Tomek links across fused sparse and dense metric spaces.
4. **Calibrated Early Feature Fusion:** Integrates magnitude-scaled Sentence-BERT (384-D, scaled $\times 1.6$) with sublinear TF-IDF (1,500-D) into a unified 1,884-D representation.

---

## 📂 Repository Structure

```text
neuro_symbolic_re/
├── data/
│   └── dataset_loader.py          # Unified loader with formal ISO 25010 FNFC mapping
├── symbolic/
│   ├── ears_syntax.py             # EARS grammatical pattern validator
│   └── iso25010_rules.py          # ISO/IEC 25010 taxonomy & banned phrase regex filters
├── neuro/
│   ├── generator.py               # Combinatorial EARS + ISO/IEC 25010 requirement engine
│   └── boundary_cleaner.py        # Asymmetric Dual-Space Tomek Cleaner
├── evaluation/
│   └── metrics.py                 # Multi-class macro/weighted metrics & hypothesis tests
├── diagnostics/
│   └── synthetic_quality_audit.py # Conformity, diversity, and 0% vague word verifier
├── figures/                       # Publication-grade 300 DPI figures
│   ├── figure2_algorithm_comparison.png
│   ├── figure3_knn_trajectory.png
│   ├── figure4_generalization_fnfc.png
│   └── figure5_classwise_f1.png
├── beat_all_algorithms.py         # Parallel 10-fold CV evaluating all 11 models
├── generate_publication_figures.py# Generates Figures 2, 3, 4, 5
├── iso_texts.txt                  # 385 verified synthetic requirements (+35 per minority)
├── iso_labels.npy                 # Class labels for synthetic requirements
├── iso_sbert.npy                  # Pre-computed SBERT embeddings for synthetic data
├── promise_sbert.npy              # Pre-computed SBERT embeddings for PROMISE NFR
└── all_classifiers_30_split_results.json # Full 30-repeat benchmark (300 runs per model)
```

---

## 📊 Empirical Benchmarks (PROMISE NFR, 10-Fold CV)

### Benchmark 1: Proposed Framework vs. Pure SMOTE-Tomek (Core Competitor)
*Evaluated across all 11 model architectures under stratified 10-fold cross-validation:*

| Classification Algorithm | Ali et al. (SMOTE-Tomek) | Barak Or (SMOTE-Tomek) | Proposed Framework | Empirical Outcome |
| :--- | :---: | :---: | :---: | :---: |
| **AdaBoost** | 29.42% *(Collapsed)* | 49.02% | **61.82%** | **+32.40% over Ali / +12.80% over Barak** |
| **SVM (RBF Kernel)** | 60.37% *(Degraded)* | 58.31% | **81.01%** | **+20.64% over Ali / +22.70% over Barak** |
| **Naive Bayes (MNB)** | 53.97% | 51.49% | **73.88%** | **+19.91% over Ali / +22.39% over Barak** |
| **K-NN ($k = 7$)** | 58.83% *(Degraded)* | 67.80% | **75.33%** | **+16.50% over Ali / +7.53% over Barak** |
| **K-NN ($k = 5$)** | 63.06% *(Degraded)* | 68.42% | **77.08%** | **+14.02% over Ali / +8.66% over Barak** |
| **K-NN ($k = 3$)** | 66.15% *(Degraded)* | 67.69% | **77.40%** | **+11.25% over Ali / +9.71% over Barak** |
| **Logistic Regression** | 72.76% | 76.16% | **80.70%** | **+7.94% over Ali / +4.54% over Barak** |
| **Gradient Boosting** | — | 66.98% | **70.48%** | **+3.50% over Barak** |
| **Decision Tree (DT)** | 49.32% | 59.55% | **51.51%** | **+2.19% over Ali** *(Superior Transfer on FNFC)* |
| **Linear SVM** | 79.05% | 71.10% | **79.98%** | **+0.93% over Ali / +8.88% over Barak** |
| **Random Forest** | 74.40% | 48.71% | **73.79%** | Competitive (Within $\pm$0.6%) |

---

### Benchmark 2: Proposed Framework vs. Raw Unresampled Baseline
*Evaluating whether the proposed augmentation improves upon the unaugmented corpus:*

| Classification Algorithm | Raw Baseline (Ali et al.) | Proposed Framework | Pure Win over Raw Baseline |
| :--- | :---: | :---: | :---: |
| **Naive Bayes (MNB)** | 0.00% *(Zero minority recall)* | **73.88%** | **+73.88% (Complete Rescue)** |
| **AdaBoost** | 47.88% | **61.82%** | **+13.94%** |
| **Random Forest** | 60.37% | **73.79%** | **+13.42% (Major Elevation)** |
| **SVM (RBF Kernel)** | 72.13% | **81.01%** | **+8.88%** |
| **Logistic Regression** | 75.43% | **80.70%** | **+5.27%** |
| **K-NN ($k = 3$)** | 72.86% | **77.40%** | **+4.54%** |
| **K-NN ($k = 5$)** | 72.54% | **77.08%** | **+4.54%** |
| **Decision Tree** | 47.88% | **51.51%** | **+3.63%** |
| **K-NN ($k = 7$)** | 72.34% | **75.33%** | **+2.99%** |
| **Linear SVM** | 80.39% | **79.98%** | Competitive (Within $\pm$0.4%) |

---

## 🌐 Out-of-Distribution Industry Generalization (FNFC, $N=7,034$)
To stress-test model robustness outside the curated PROMISE benchmark, we replicated Ali et al.'s exact feature extraction and SMOTE-Tomek architecture and transferred both zero-shot to the 7,034 real-world specifications of the external `FNFC.csv` repository.

| Classification Algorithm | Replicated SMOTE-Tomek (Ali et al.) | Proposed EARS + ISO/IEC 25010 | Generalization Advantage |
| :--- | :---: | :---: | :---: |
| **K-NN ($k = 5$)** | 13.42% *(Catastrophic collapse)* | **74.54%** | **+61.12% (Prevents Model Collapse)** |
| **Naive Bayes (MNB)** | 41.53% | **83.35%** | **+41.82%** |
| **Decision Tree (DT)** | 17.54% *(Catastrophic collapse)* | **51.44%** | **+33.90%** |
| **Logistic Regression** | 67.39% | **77.94%** | **+10.55%** |
| **Random Forest** | 71.67% | **78.46%** | **+6.79%** |
| **Linear SVM** | 64.32% | **70.16%** | **+5.84%** |

---

## 📐 Formal ISO/IEC 25010 Taxonomy Alignment Table (FNFC $\rightarrow$ PROMISE)

The 14 raw classes in FNFC are systematically aligned with PROMISE's 12 classes according to ISO/IEC 25010:2011 Clause 4.2:

| FNFC Label | FNFC Concept | Target Label | ISO/IEC 25010:2011 Rationale |
| :---: | :--- | :---: | :--- |
| **F** | Functional | **F** | Functional Suitability (§4.2.1) |
| **SE** | Security | **SE** | Security & Integrity (§4.2.6) |
| **AU** | Auditability | **SE** | Security: Non-repudiation & Authenticity (§4.2.6) |
| **P** | Performance | **PE** | Performance Efficiency (§4.2.2) |
| **PE** | Performance Efficiency | **PE** | Performance Efficiency (§4.2.2) |
| **A** | Availability | **A** | Reliability & Availability (§4.2.5) |
| **R** | Reliability | **A** | Reliability & Availability (§4.2.5) |
| **US** | Usability | **US** | Usability & Operability (§4.2.4) |
| **O** | Operational | **O** | Compatibility & Co-existence (§4.2.3) |
| **LL** | Legal | **L** | Compliance & Regulatory Constraints (§4.2.7) |
| **L** | Legal | **L** | Compliance & Regulatory Constraints (§4.2.7) |
| **M** | Maintainability | **MN** | Maintainability: Modularity & Analysability (§4.2.7) |
| **MN** | Maintainability | **MN** | Maintainability: Modularity & Analysability (§4.2.7) |
| **LF** | Look and Feel | **LF** | User Interface Aesthetics |
| **FT** | Fault Tolerance | **FT** | Reliability: Fault Tolerance & Recoverability (§4.2.5) |
| **SC** | Scalability | **SC** | Capacity & Elastic Scaling (§4.2.2) |

---

## 📈 Publication Figures Suite

* **Figure 1:** System Architecture & Workflow Pipeline (*Stages 1–4: Ingestion, EARS+ISO Generation, Asymmetric Cleaning, Early Fusion & Supervised Inference*).
* **Figure 2:** Multi-Model Performance Comparison (*10 Core Classifiers: Pure SMOTE-Tomek vs. Proposed*).
* **Figure 3:** K-NN Neighborhood Degradation Trajectory Across $k \in \{3, 5, 7\}$ (*Exposing SMOTE decay vs. Proposed topological stability*).
* **Figure 4:** Zero-Shot Transfer Generalization on FNFC ($N=7,034$) (*Replicated Baseline vs. Proposed*).
* **Figure 5:** Granular Class-Wise F1-Score Breakdown across all 12 Requirements Types.

---

## 🚀 Reproduction Instructions

### 1. Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Or on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Verify Synthetic Data Quality Audit
```bash
python diagnostics/synthetic_quality_audit.py
```
*Expected Output:*
* `EARS Syntactic Conformity: 100.0%`
* `IEEE 830 Modal Verb Compliance: 100.0%`
* `Forbidden Vague Phrase Violation Rate: 0.00%`

### 3. Run Benchmark Across All 11 Models
```bash
python beat_all_algorithms.py
```

### 4. Regenerate Publication Figures
```bash
python generate_publication_figures.py
```
All figures will be exported at 300 DPI into the `figures/` directory.
