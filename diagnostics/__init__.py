"""Quality diagnostics and baseline comparative audit package."""
from .synthetic_quality_audit import audit_synthetic_corpus
from .eda_synonym_baseline import run_eda_baseline_audit

__all__ = ["audit_synthetic_corpus", "run_eda_baseline_audit"]
