"""Unit tests for the Asymmetric Dual-Space Boundary Cleaner."""

import unittest
import numpy as np
from neuro.boundary_cleaner import AsymmetricDualSpaceBoundaryCleaner


class TestBoundaryCleaner(unittest.TestCase):
    def test_asymmetric_cleaner_locks_human_points(self):
        cleaner = AsymmetricDualSpaceBoundaryCleaner(alpha=1.6, max_features=100, random_state=42)
        
        real_texts = ["The system shall process requests.", "The system shall encrypt data."]
        real_labels = ["PE", "SE"]
        synth_texts = ["The system shall process quickly.", "The system shall authenticate data."]
        synth_labels = ["PE", "SE"]
        
        real_sbert = np.random.randn(2, 384)
        synth_sbert = np.random.randn(2, 384)
        
        cleaned_t, cleaned_l, cleaned_sb, n_pruned = cleaner.clean_synthetic_only(
            real_texts, real_labels, synth_texts, synth_labels, real_sbert, synth_sbert
        )
        
        # Crucial property: All authentic human data must be strictly preserved
        self.assertGreaterEqual(len(cleaned_t), len(real_texts))
        self.assertEqual(cleaned_t[0], real_texts[0])
        self.assertEqual(cleaned_t[1], real_texts[1])
        self.assertIsInstance(n_pruned, int)


if __name__ == "__main__":
    unittest.main()
