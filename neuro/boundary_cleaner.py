"""
Asymmetric Dual-Space Boundary Cleaner.
Ensures that authentic human-written training requirements are NEVER pruned (100% locked).
Operates in true Dual-Space (concatenated SBERT dense semantic embeddings + TF-IDF sparse lexical tokens).
Only synthetic requirements that form Tomek link overlaps with opposing classes
are pruned, preserving true ground-truth anchors while removing synthetic noise.
"""

import numpy as np
from typing import Tuple, List, Optional
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


class AsymmetricDualSpaceBoundaryCleaner:
    """
    Cleans synthetic boundary artifacts in unified dual-space (Dense SBERT + Sparse TF-IDF)
    while strictly protecting authentic ground-truth data.
    """
    def __init__(self, alpha: float = 1.6, max_features: int = 1200, random_state: int = 42):
        self.alpha = alpha
        self.max_features = max_features
        self.random_state = random_state
        self.vectorizer = None

    def clean_synthetic_only(
        self,
        real_texts: List[str],
        real_labels: List[str],
        synth_texts: List[str],
        synth_labels: List[str],
        real_embeddings: Optional[np.ndarray] = None,
        synth_embeddings: Optional[np.ndarray] = None
    ) -> Tuple[List[str], List[str], Optional[np.ndarray], int]:
        """
        Takes real and synthetic data separately (both text and optional dense embeddings).
        Constructs the unified dual-space manifold.
        Identifies if any synthetic sample is in a mutual Tomek pair with an opposite-class sample.
        Prunes strictly only ambiguous synthetic points, guaranteeing zero modification to real points.

        Returns:
            (final_texts, final_labels, final_embeddings, num_pruned)
        """
        if len(synth_texts) == 0:
            return real_texts, real_labels, real_embeddings, 0

        all_texts = list(real_texts) + list(synth_texts)
        all_labels = list(real_labels) + list(synth_labels)
        n_real = len(real_texts)

        # 1. Sparse Lexical Representation (TF-IDF)
        self.vectorizer = TfidfVectorizer(max_features=self.max_features, ngram_range=(1, 2), stop_words='english', sublinear_tf=True)
        X_tfidf = self.vectorizer.fit_transform(all_texts).toarray()
        X_tfidf_norm = normalize(X_tfidf)

        # 2. Dual-Space Construction: Dense SBERT + Sparse TF-IDF
        if real_embeddings is not None and synth_embeddings is not None:
            dense_all = np.vstack([real_embeddings, synth_embeddings])
            dense_norm = normalize(dense_all) * self.alpha
            X_dual = np.hstack([dense_norm, X_tfidf_norm])
        else:
            X_dual = X_tfidf_norm

        y = np.array(all_labels)

        # 3. Fit Nearest Neighbors in Dual Space (k=2: self + 1 nearest neighbor)
        nn = NearestNeighbors(n_neighbors=2, metric='cosine')
        nn.fit(X_dual)
        distances, indices = nn.kneighbors(X_dual)

        # 4. Asymmetric Tomek Link Condition: Prune ONLY synthetic points
        kept_synth_indices = []
        num_pruned = 0

        for i in range(len(synth_texts)):
            global_idx = n_real + i
            neighbor_idx = indices[global_idx, 1]
            neighbor_label = y[neighbor_idx]
            current_label = y[global_idx]

            # If the nearest neighbor has a different class, check mutual condition
            if neighbor_label != current_label:
                reverse_neighbor = indices[neighbor_idx, 1]
                if reverse_neighbor == global_idx:
                    # Mutual nearest neighbor of opposite class (Tomek link) -> Prune synthetic point!
                    num_pruned += 1
                    continue

            kept_synth_indices.append(i)

        final_texts = list(real_texts) + [synth_texts[i] for i in kept_synth_indices]
        final_labels = list(real_labels) + [synth_labels[i] for i in kept_synth_indices]

        if real_embeddings is not None and synth_embeddings is not None:
            kept_synth_emb = synth_embeddings[kept_synth_indices] if len(kept_synth_indices) > 0 else np.empty((0, real_embeddings.shape[1]))
            final_embeddings = np.vstack([real_embeddings, kept_synth_emb])
        else:
            final_embeddings = None

        return final_texts, final_labels, final_embeddings, num_pruned
