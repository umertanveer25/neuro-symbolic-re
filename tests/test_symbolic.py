"""Unit tests for symbolic rules, EARS syntax, and ISO 25010 validation."""

import unittest
from symbolic.ears_syntax import analyze_ears_pattern
from symbolic.iso25010_rules import validate_iso_quality_rules


class TestSymbolicRules(unittest.TestCase):
    def test_ubiquitous_ears_pattern(self):
        spec = "The database engine shall maintain memory utilization below ninety percent."
        res = analyze_ears_pattern(spec)
        self.assertTrue(res["is_ears_compliant"])
        self.assertEqual(res["pattern"], "ubiquitous")

    def test_event_driven_ears_pattern(self):
        spec = "When an invalid credential occurs, the system shall lock the account."
        res = analyze_ears_pattern(spec)
        self.assertTrue(res["is_ears_compliant"])
        self.assertEqual(res["pattern"], "event_driven")

    def test_unwanted_behavior_ears_pattern(self):
        spec = "If primary network connectivity fails, then the system shall switch to backup."
        res = analyze_ears_pattern(spec)
        self.assertTrue(res["is_ears_compliant"])
        self.assertEqual(res["pattern"], "unwanted_behavior")

    def test_non_ears_specification(self):
        spec = "System needs to be user friendly and fast."
        res = analyze_ears_pattern(spec)
        self.assertFalse(res["is_ears_compliant"])

    def test_iso_quality_rules_valid_security(self):
        spec = "The authentication service shall encrypt all passwords using AES-256."
        is_valid, _ = validate_iso_quality_rules(spec, "SE")
        self.assertTrue(is_valid)

    def test_iso_quality_rules_forbidden_vague_terms(self):
        spec = "The system shall process requests as fast as possible."
        is_valid, reasons = validate_iso_quality_rules(spec, "PE")
        self.assertFalse(is_valid)
        self.assertTrue(any("vague" in r.lower() for r in reasons))

    def test_iso_quality_rules_missing_modal(self):
        spec = "The system encrypts passwords using AES-256."
        is_valid, reasons = validate_iso_quality_rules(spec, "SE")
        self.assertFalse(is_valid)
        self.assertTrue(any("modal" in r.lower() for r in reasons))


if __name__ == "__main__":
    unittest.main()
