#!/usr/bin/env python3
"""
Quick baseline evaluation to identify middle-performing subjects.
Dean's instructions: "Take subjects that are somewhere in middle in terms of metrics"
"""

import os
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path

BASELINE_CONFIG = 'config/bracelet/filter_comparison_baseline.json'
ALL_SUBJECTS = list(range(1, 11))  # ID01 through ID10
PYTHON_PATH = '/home/abed/miniconda3/envs/imugr/bin/python3'


def quick_eval_subject(subject_id):
    """Run quick evaluation for one subject."""
    print(f"Evaluating subject {subject_id}...")

    cmd = [
        PYTHON_PATH, 'trainer/train_conv.py',
        '--json', BASELINE_CONFIG,
        '--loo', str(subject_id)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  ERROR: Failed for subject {subject_id}")
        print(f"  STDERR: {result.stderr[:500]}")  # Print first 500 chars of error
        print(f"  STDOUT: {result.stdout[:500]}")  # Print first 500 chars of output
        return None

    # Read metrics - the training script adds extra suffix to output dir
    output_dir = f"outputs/filter_loo_baseline_s{subject_id:02d}_baseline"
    metrics_path = Path(output_dir) / 'metrics.csv'

    if metrics_path.exists():
        metrics_df = pd.read_csv(metrics_path)
        final_metrics = metrics_df.iloc[-1]

        return {
            'subject': subject_id,
            'accuracy': final_metrics['Accuracy'],
            'recall': final_metrics['Recall'],
            'precision': final_metrics['Precision']
        }
    else:
        print(f"  WARNING: Metrics not found for subject {subject_id}")
        return None


def main():
    """Evaluate all subjects and identify middle performers."""
    print("="*80)
    print("IDENTIFYING MIDDLE-PERFORMING SUBJECTS")
    print("="*80)

    results = []
    for subject_id in ALL_SUBJECTS:
        result = quick_eval_subject(subject_id)
        if result is not None:
            results.append(result)
            print(f"  Subject {subject_id}: Acc={result['accuracy']:.3f}, "
                  f"Rec={result['recall']:.3f}, Prec={result['precision']:.3f}")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('outputs/baseline_subject_performance.csv', index=False)

    # Sort by accuracy to find middle performers
    results_df = results_df.sort_values('accuracy')

    print("\n" + "="*80)
    print("SUBJECT RANKING BY ACCURACY")
    print("="*80)
    for idx, row in results_df.iterrows():
        print(f"  Subject {row['subject']:2d}: {row['accuracy']:.3f}")

    # Select middle 3 subjects
    n_subjects = len(results_df)
    middle_start = (n_subjects - 3) // 2
    middle_subjects = results_df.iloc[middle_start:middle_start+3]['subject'].tolist()

    print("\n" + "="*80)
    print("RECOMMENDED MIDDLE-PERFORMING SUBJECTS")
    print("="*80)
    print(f"  Subjects: {middle_subjects}")
    print(f"\nUpdate TEST_SUBJECTS in run_filter_loo_comparison.py to: {middle_subjects}")

    return middle_subjects


if __name__ == '__main__':
    main()
