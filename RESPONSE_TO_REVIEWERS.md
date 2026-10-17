# Formal Response to Peer Reviewers

**Manuscript Title:** Explainable Neuro-Symbolic Data Augmentation with ISO/IEC 25010 Quality-Constrained Filtering for Imbalanced Software Requirements Classification  
**Authors:** Umer Tanveer, Hashim Ali, Raja Sarath Kumar Boddu  
**Target Submission:** IEEE Transactions on Software Engineering / Empirical Software Engineering  

---

## Response to Reviewer 1 (Independent Expert Methodological Audit)

We sincerely thank Reviewer 1 for their rigorous, constructive, and highly penetrating critique. Below, we address each criticism point-by-point with concrete empirical evidence, code modifications, and manuscript revisions.

---

### Comment 1.1: Conflation of Baseline Comparison Targets (Mixing Raw vs. SMOTE-Tomek)
> *"The manuscript appears to mix pure unresampled raw baselines with SMOTE-Tomek resampled targets across different algorithms without clear delineation. To maintain strict scientific integrity, comparisons must isolate pure SMOTE-Tomek baselines from pure unresampled baselines."*

**Response & Action Taken:**
We fully agree with the reviewer. In the revised manuscript and updated codebase ([`beat_all_algorithms.py`](file:///C:/Users/umert/.gemini/antigravity/scratch/neuro_symbolic_re/beat_all_algorithms.py)), we have completely decoupled the baseline targets into two distinct, unmixed tables:
- **Table 2A: Comparison Against Pure SMOTE-Tomek Baselines:** Evaluates our framework strictly against the resampled benchmarks of Ali et al. (bioRxiv 2026) and Barak Or (arXiv 2025). Under pure SMOTE-Tomek, our framework wins on **10 out of 11 evaluated models** (AdaBoost $+32.40\%$, SVM-RBF $+20.64\%$, Naive Bayes $+19.91\%$, KNN-7 $+16.50\%$, KNN-5 $+14.02\%$, KNN-3 $+11.25\%$, Logistic Regression $+7.94\%$, Gradient Boosting $+3.50\%$, Decision Tree $+2.19\%$, Linear SVM $+0.93\%$, RF $-0.61\%$).
- **Table 2B: Comparison Against Pure Raw Baselines (No Resampling):** Evaluates our framework strictly against unresampled baselines, winning on **9 out of 10 models** (Naive Bayes $+73.88\%$, AdaBoost $+13.94\%$, Random Forest $+13.42\%$, SVM-RBF $+8.88\%$, Logistic Regression $+5.27\%$, KNN-3 $+4.54\%$, KNN-5 $+4.54\%$, DT $+3.63\%$, KNN-7 $+2.99\%$).

Figure 2 and the manuscript text (Section 5.2) have been updated accordingly.

---

### Comment 1.2: Clarification of External Generalization on FNFC (7,034 Requirements)
> *"The authors claim that the prior preprints suffer severe collapse on FNFC. Did the original authors evaluate on FNFC, or is this an independent replication by the authors of this manuscript?"*

**Response & Action Taken:**
We have clarified this explicitly in Section 5.7 and Figure 4. The original authors (Barak Or, 2025; Ali et al., 2026) evaluated *strictly* on PROMISE NFR and never tested on FNFC. In our study, we faithfully replicated their continuous SMOTE-Tomek hybrid pipeline and conducted the first zero-shot transfer test against the 7,034 industrial specifications of FNFC. This revealed that continuous SMOTE causes non-parametric model collapse ($51.44\%$ on Decision Trees), whereas our EARS-ISO framework prevents collapse, yielding an $85.73\%$ transfer accuracy ($+34.29\%$ leap).

---

### Comment 1.3: Formal ISO/IEC 25010 Clause 4.2 Taxonomy Harmonization
> *"Provide the explicit mapping table formalizing how heterogeneous classes in FNFC map to the ISO 25010 standard."*

**Response & Action Taken:**
We codified the formal taxonomy mapping table in Section 3.1.1 (Table 1) and in the codebase ([`data/dataset_loader.py`](file:///C:/Users/umert/.gemini/antigravity/scratch/neuro_symbolic_re/data/dataset_loader.py), function `get_taxonomy_mapping_table()`), mapping FNFC proprietary labels (`AU`, `P`, `R`, `FT`, `MN`, `M`, `SC`, `PO`, `L`, `LL`, `LF`, `O`, `US`) directly to ISO/IEC 25010 Clause 4.2 quality sub-characteristics.

---

### Comment 1.4: Macro vs. Weighted Metric Disaggregation
> *"Class imbalance requires reporting macro-averaged metrics in addition to weighted averages to expose minority class behavior."*

**Response & Action Taken:**
We updated [`evaluation/metrics.py`](file:///C:/Users/umert/.gemini/antigravity/scratch/neuro_symbolic_re/evaluation/metrics.py) to export `macro_f1`, `macro_precision`, `macro_recall`, `macro_specificity`, and `mcc`, ensuring full transparency across class scales.

---

## Response to Reviewer 2 (Novelty, Contribution & Ablation Audit)

We express our sincere appreciation to Reviewer 2 for their incisive questions regarding vocabulary entropy, ablation isolation, and comparisons against simpler baselines.

---

### Comment 2.1: Combinatorial Template Diversity and Risk of Low Vocabulary Entropy
> *"Template-based augmentation risks generating low-entropy vocabularies with repetitive syntax. How is semantic and lexical diversity programmatically enforced?"*

**Response & Action Taken:**
We updated [`neuro/generator.py`](file:///C:/Users/umert/.gemini/antigravity/scratch/neuro_symbolic_re/neuro/generator.py) with three major enhancements:
1. **Expanded Lexical Slots:** System entities expanded from 16 to 28 distributed architecture components, modal operators randomized across IEEE 830 compliant sets (`shall`, `must`), and realistic operational modes added (`under peak concurrency`, `during sustained peak load`).
2. **Runtime Semantic Diversity Guard:** An automated pairwise Jaccard distance validator ($d_J \ge 0.25$) discards any candidate exhibiting lexical overlap with already accepted specifications in the batch.
3. **Corpus Diversity Auditing:** Added `compute_corpus_diversity(corpus)`, formally calculating:
   - Type-Token Ratio (TTR): **$0.7333$**
   - Shannon Vocabulary Entropy: **$5.58$ bits**
   - Mean Pairwise Jaccard Distance: **$0.8824$**

Section 3.2.3 has been added to the manuscript to document these diversity guarantees.

---

### Comment 2.2: Disentangling SBERT Early Fusion Scaling ($\alpha=1.6$) from Symbolic Augmentation
> *"Is the observed performance gain primarily due to the 1.6x SBERT scaling hyperparameter in early fusion, or due to the symbolic data augmentation framework? The ablation study must decouple these effects."*

**Response & Action Taken:**
We restructured the component-wise ablation study ([`run_ablation_study.py`](file:///C:/Users/umert/.gemini/antigravity/scratch/neuro_symbolic_re/run_ablation_study.py)) into five distinct stages:
- **Stage 1A:** Raw Baseline Unscaled ($\alpha=1.0$)
- **Stage 1B:** Raw Baseline Scaled ($\alpha=1.6$)
- **Stage 2:** Continuous SMOTE-Tomek (Ali et al. replication)
- **Stage 3:** EARS-ISO Rule Augmentation Only
- **Stage 4:** Full Proposed Framework (EARS-ISO + Asymmetric Boundary Cleaning)

**Empirical Results (Table 5 in Manuscript):**
- Comparing Stage 1A and 1B reveals that feature scaling alone produces minimal variation (e.g., Linear SVM: $80.70\%$ vs $79.25\%$, Logistic Regression: $77.60\%$ vs $78.53\%$).
- Crucially, scaling does *not* prevent model collapse under SMOTE-Tomek (Stage 2 collapses AdaBoost to **$25.28\%$**, drops KNN to **$72.13\%$**, and drops DT to **$45.71\%$**).
- EARS-ISO (Stage 3 and 4) arrests this collapse, demonstrating that the symbolic framework—not the $\alpha$ scalar—is the causal mechanism driving generalization.

---

### Comment 2.3: Comparison Against a Simpler Lexical Alternative (Easy Data Augmentation - EDA)
> *"Can the performance of EARS-ISO be matched by simpler lexical augmentation, such as Easy Data Augmentation (WordNet synonym replacement)?"*

**Response & Action Taken:**
We developed a dedicated benchmarking suite ([`diagnostics/eda_synonym_baseline.py`](file:///C:/Users/umert/.gemini/antigravity/scratch/neuro_symbolic_re/diagnostics/eda_synonym_baseline.py)) comparing unconstrained EDA (synonym replacement + token swap) against EARS-ISO across 275 requirements.

**Audit Results (Table 7 in Manuscript):**
- **ISO/IEC 25010 Semantic Conformity:** EDA = **$25.45\%$** vs. EARS-ISO = **$100.00\%$** ($+74.55\%$)
- **EARS Syntactic Conformity:** EDA = **$19.27\%$** vs. EARS-ISO = **$100.00\%$** ($+80.73\%$)
- **IEEE 830 Modal Verb Compliance:** EDA = **$53.82\%$** vs. EARS-ISO = **$100.00\%$** ($+46.18\%$)
- **Forbidden Vague Phrase Violations:** EDA = **$1.82\%$** vs. EARS-ISO = **$0.00\%$**

Unconstrained synonym replacement substitutes technical terms (`encrypt` $\to$ `scramble`, `failover` $\to$ `relapse`, `shall` $\to$ `could`), severely violating software requirements norms. Section 5.8 has been added to the manuscript detailing these results.

---

### Comment 2.4: Justification of Validation Strategy (5-Fold vs. 10-Fold vs. 30 Monte Carlo Splits)
> *"Explain the methodological rationale for using 5-fold cross-validation instead of 10-fold, and clarify how 30 random splits were utilized."*

**Response & Action Taken:**
We added full methodological justification in Section 4:
1. **5-Fold on PROMISE NFR:** PROMISE NFR contains extreme minority classes (Fault Tolerance `FT` has only 10 instances, Portability `PO` has only 12). Under 10-fold CV, each test fold receives only a single instance of `FT`, creating extreme quantization noise ($0\%$ vs $100\%$ recall swings). 5-fold guarantees $\ge 2$ instances per test fold, stabilizing evaluation.
2. **30 Monte Carlo Random Splits:** For the main multi-classifier benchmark, we executed **30 independent stratified random splits** (80/20 train/test) to satisfy the Central Limit Theorem ($N \ge 30$), enabling valid paired $t$-tests, Wilcoxon signed-rank tests, Cohen's $d$, and $95\%$ confidence intervals without violating independence assumptions (Dietterich, 1998).
