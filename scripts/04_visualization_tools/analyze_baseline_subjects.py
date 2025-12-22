#!/usr/bin/env python3
# save as: analyze_baseline_subjects.py

import pandas as pd
from pathlib import Path

results = []
for subject in range(1, 11):
    metrics_path = Path(f"outputs/filter_loo_baseline_s{subject:02d}_baseline/metrics.csv")
    if metrics_path.exists():
        df = pd.read_csv(metrics_path)
        final = df.iloc[-1]
        results.append({
            'subject': subject,
            'accuracy': final['Accuracy'],
            'recall': final['Recall'],
            'precision': final['Precision']
        })

# Create dataframe and sort
df = pd.DataFrame(results).sort_values('accuracy')

print("="*60)
print("BASELINE SUBJECT RANKING (worst → best)")
print("="*60)
for _, row in df.iterrows():
    print(f"  Subject {row['subject']:2.0f}: Acc={row['accuracy']:.4f}, Rec={row['recall']:.4f}, Prec={row['precision']:.4f}")

# Find middle 3
n = len(df)
middle_start = (n - 3) // 2
middle_subjects = df.iloc[middle_start:middle_start+3]['subject'].astype(int).tolist()

print("\n" + "="*60)
print(f"MIDDLE PERFORMERS (for filter testing): {middle_subjects}")
print("="*60)

# Save full results
df.to_csv('outputs/baseline_subject_ranking.csv', index=False)
print(f"\nFull results saved to: outputs/baseline_subject_ranking.csv")