"""Unit tests for the neuro-symbolic generator and diversity guard."""

import unittest
from neuro.generator import EnhancedNeuroSymbolicAugmenter


class TestNeuroGenerator(unittest.TestCase):
    def test_generator_candidate_creation(self):
        augmenter = EnhancedNeuroSymbolicAugmenter(random_seed=42)
        candidate = augmenter.generate_candidate("PE")
        self.assertIsInstance(candidate, str)
        self.assertGreater(len(candidate), 20)
        self.assertTrue(any(m in candidate.lower() for m in ["shall", "must"]))

    def test_generator_verified_batch(self):
        augmenter = EnhancedNeuroSymbolicAugmenter(random_seed=42)
        batch, stats = augmenter.generate_verified_batch("SE", n_needed=5)
        self.assertEqual(len(batch), 5)
        self.assertEqual(stats["passed"], 5)
        self.assertGreaterEqual(stats["attempted"], 5)

    def test_corpus_diversity_metrics(self):
        corpus = [
            "The security module shall encrypt user credentials within 50 milliseconds.",
            "When an invalid credential occurs, the system shall lock the account.",
            "While under ninety percent server workload, the database engine must maintain throughput."
        ]
        metrics = EnhancedNeuroSymbolicAugmenter.compute_corpus_diversity(corpus)
        self.assertIn("ttr", metrics)
        self.assertIn("shannon_entropy", metrics)
        self.assertIn("mean_jaccard_distance", metrics)
        self.assertGreater(metrics["ttr"], 0.5)
        self.assertGreater(metrics["shannon_entropy"], 3.0)


if __name__ == "__main__":
    unittest.main()
