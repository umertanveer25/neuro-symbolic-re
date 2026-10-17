"""Unit tests for multi-class and macro-averaged metrics."""

import unittest
from evaluation.metrics import calculate_comprehensive_metrics


class TestMetrics(unittest.TestCase):
    def test_calculate_comprehensive_metrics(self):
        y_true = ["PE", "SE", "US", "A", "FT"]
        y_pred = ["PE", "SE", "US", "A", "SE"]  # 4 correct, 1 wrong
        
        metrics = calculate_comprehensive_metrics(y_true, y_pred)
        self.assertIn("accuracy", metrics)
        self.assertIn("macro_f1", metrics)
        self.assertIn("f1_score", metrics)
        self.assertIn("macro_precision", metrics)
        self.assertIn("macro_recall", metrics)
        self.assertIn("mcc", metrics)
        
        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertLessEqual(metrics["accuracy"], 1.0)
        self.assertGreaterEqual(metrics["macro_f1"], 0.0)
        self.assertLessEqual(metrics["macro_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
