#!/usr/bin/env python
"""
Collect and summarize experiment results for paper tables.
Usage: python scripts/collect_results.py --log_dir logs/experiments_xxx
"""

import os
import re
import json
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np


def parse_log_file(log_path):
    """Parse training log to extract final metrics."""
    metrics = {}

    with open(log_path, 'r') as f:
        content = f.read()

    # Find the last test results table
    # Pattern matches metric values like "mAUROC_sp_max: 95.123"
    patterns = {
        'mAUROC_sp': r'mAUROC_sp_max[^\d]*(\d+\.?\d*)',
        'mAUROC_px': r'mAUROC_px[^\d]*(\d+\.?\d*)',
        'mAUPRO_px': r'mAUPRO_px[^\d]*(\d+\.?\d*)',
    }

    for metric_name, pattern in patterns.items():
        matches = re.findall(pattern, content)
        if matches:
            # Take the last occurrence (final epoch)
            metrics[metric_name] = float(matches[-1])

    return metrics


def collect_main_results(log_dir):
    """Collect main results from benchmark datasets."""
    print("\n" + "="*60)
    print("MAIN RESULTS - Benchmark Datasets")
    print("="*60)

    datasets = ['mvtec', 'visa', 'btad', 'realiad']
    results = {}

    for dataset in datasets:
        log_file = Path(log_dir) / f"{dataset}_train.log"
        if log_file.exists():
            metrics = parse_log_file(log_file)
            results[dataset] = metrics
            print(f"\n{dataset.upper()}:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.2f}")

    return results


def collect_ablation_results(log_dir):
    """Collect ablation study results."""
    print("\n" + "="*60)
    print("ABLATION STUDY - Loss Components")
    print("="*60)

    ablations = {
        'CosLoss only': 'ablation_cos_only.log',
        'CosLoss + Dense': 'ablation_cos_dense.log',
        'CosLoss + Proto': 'ablation_cos_proto.log',
        'Full (Ours)': 'ablation_full.log',
    }

    results = {}
    for name, log_file in ablations.items():
        log_path = Path(log_dir) / log_file
        if log_path.exists():
            metrics = parse_log_file(log_path)
            results[name] = metrics
            print(f"\n{name}:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.2f}")

    return results


def collect_sensitivity_results(log_dir, prefix, param_name, values):
    """Collect sensitivity analysis results."""
    print(f"\n" + "="*60)
    print(f"SENSITIVITY - {param_name}")
    print("="*60)

    results = {}
    for val in values:
        log_file = Path(log_dir) / f"{prefix}_{val}.log"
        if log_file.exists():
            metrics = parse_log_file(log_file)
            results[val] = metrics
            print(f"\n{param_name}={val}:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.2f}")

    return results


def collect_seed_results(log_dir):
    """Collect multi-seed results for statistical significance."""
    print("\n" + "="*60)
    print("STATISTICAL SIGNIFICANCE - Multiple Seeds")
    print("="*60)

    seeds = [42, 123, 456, 789, 1024]
    all_metrics = defaultdict(list)

    for seed in seeds:
        log_file = Path(log_dir) / f"seed_{seed}.log"
        if log_file.exists():
            metrics = parse_log_file(log_file)
            for k, v in metrics.items():
                all_metrics[k].append(v)

    print("\nResults (mean +/- std):")
    summary = {}
    for metric, values in all_metrics.items():
        if values:
            mean = np.mean(values)
            std = np.std(values)
            print(f"  {metric}: {mean:.2f} +/- {std:.2f}")
            summary[metric] = {'mean': mean, 'std': std, 'values': values}

    return summary


def generate_latex_table(results, caption, label):
    """Generate LaTeX table from results."""
    if not results:
        return ""

    # Get all metrics
    all_metrics = set()
    for metrics in results.values():
        all_metrics.update(metrics.keys())
    all_metrics = sorted(all_metrics)

    # Build table
    header = " & ".join(["Method"] + all_metrics) + " \\\\"
    rows = []
    for name, metrics in results.items():
        row_values = [name]
        for m in all_metrics:
            if m in metrics:
                row_values.append(f"{metrics[m]:.2f}")
            else:
                row_values.append("-")
        rows.append(" & ".join(row_values) + " \\\\")

    latex = f"""
\\begin{{table}}[h]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\begin{{tabular}}{{l{'c' * len(all_metrics)}}}
\\toprule
{header}
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    return latex


def main():
    parser = argparse.ArgumentParser(description='Collect experiment results')
    parser.add_argument('--log_dir', type=str, required=True,
                        help='Directory containing experiment logs')
    parser.add_argument('--output', type=str, default='results_summary.json',
                        help='Output JSON file for results')
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"Error: Log directory {log_dir} does not exist")
        return

    all_results = {}

    # Collect all results
    all_results['main'] = collect_main_results(log_dir)
    all_results['ablation'] = collect_ablation_results(log_dir)
    all_results['n_prototypes'] = collect_sensitivity_results(
        log_dir, 'sensitivity_nproto', 'n_prototypes', [3, 5, 7, 10, 15])
    all_results['temperature'] = collect_sensitivity_results(
        log_dir, 'sensitivity_temp', 'temperature', [0.03, 0.05, 0.07, 0.1, 0.2])
    all_results['seeds'] = collect_seed_results(log_dir)

    # Save to JSON
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    # Generate LaTeX tables
    print("\n" + "="*60)
    print("LATEX TABLES")
    print("="*60)

    if all_results['main']:
        print(generate_latex_table(
            all_results['main'],
            "Main results on benchmark datasets",
            "tab:main_results"))

    if all_results['ablation']:
        print(generate_latex_table(
            all_results['ablation'],
            "Ablation study on loss components",
            "tab:ablation"))


if __name__ == '__main__':
    main()
