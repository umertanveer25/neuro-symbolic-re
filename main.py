#!/usr/bin/env python3
"""
Explainable Neuro-Symbolic Requirements Classification
Main Unified CLI Entrypoint.

Usage:
    python main.py audit         # Run synthetic requirements quality & diversity audit
    python main.py eda           # Run unconstrained EDA vs EARS-ISO baseline comparison
    python main.py ablate        # Run 5-stage component-wise ablation study
    python main.py benchmark     # Run full 11-model 30-split benchmark
    python main.py figures       # Regenerate all 5 publication figures at 300 DPI
"""

import sys
import argparse
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def run_audit_command(args):
    from diagnostics.synthetic_quality_audit import audit_synthetic_corpus
    base_dir = os.path.dirname(os.path.abspath(__file__))
    texts_path = os.path.join(base_dir, 'iso_texts.txt')
    labels_path = os.path.join(base_dir, 'iso_labels.npy')
    sbert_path = os.path.join(base_dir, 'iso_sbert.npy')
    audit_synthetic_corpus(texts_path, labels_path, sbert_path)


def run_eda_command(args):
    from diagnostics.eda_synonym_baseline import run_eda_baseline_audit
    run_eda_baseline_audit(n_samples_per_class=args.samples, seed=args.seed)


def run_ablate_command(args):
    from run_ablation_study import run_ablation
    run_ablation(n_splits=args.splits, random_state=args.seed)


def run_benchmark_command(args):
    import subprocess
    print("Launching complete 11-model 30-split Monte Carlo benchmark...")
    cmd = [sys.executable, "beat_all_algorithms.py"]
    subprocess.run(cmd, check=True)


def run_figures_command(args):
    import subprocess
    print("Regenerating all 300 DPI publication figures...")
    cmd = [sys.executable, "generate_publication_figures.py"]
    subprocess.run(cmd, check=True)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Explainable Neuro-Symbolic Requirements Classification CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Available execution subcommands")

    # Audit subcommand
    audit_parser = subparsers.add_parser("audit", help="Run synthetic data quality and diversity audit")
    audit_parser.set_defaults(func=run_audit_command)

    # EDA subcommand
    eda_parser = subparsers.add_parser("eda", help="Run unconstrained EDA vs EARS-ISO baseline audit")
    eda_parser.add_argument("--samples", type=int, default=25, help="Number of synthetic samples per minority class")
    eda_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    eda_parser.set_defaults(func=run_eda_command)

    # Ablation subcommand
    ablate_parser = subparsers.add_parser("ablate", help="Run 5-stage component-wise ablation study")
    ablate_parser.add_argument("--splits", type=int, default=5, help="Number of cross-validation folds")
    ablate_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    ablate_parser.set_defaults(func=run_ablate_command)

    # Benchmark subcommand
    bench_parser = subparsers.add_parser("benchmark", help="Run full 11-model 30-split Monte Carlo benchmark")
    bench_parser.set_defaults(func=run_benchmark_command)

    # Figures subcommand
    fig_parser = subparsers.add_parser("figures", help="Regenerate all publication figures at 300 DPI")
    fig_parser.set_defaults(func=run_figures_command)

    return parser


def cli():
    parser = build_parser()
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    cli()
