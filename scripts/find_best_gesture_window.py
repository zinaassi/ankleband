#!/usr/bin/env python3
"""
Scan all gesture regions in Subject 2 data to find one where:
  - Baseline predicts Rest for > 60% of windows inside the gesture
  - EMA model predicts the correct gesture class for > 60% of windows inside the gesture
Prints the top candidates so we can pick the cleanest one.
"""

import os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trainer.models.conv1d_model import Conv1DNet
import tables
import warnings
warnings.filterwarnings('ignore')

CLASS_NAMES = ['Rest', 'Grasp', 'Pinch', 'Rotate CW', 'Rotate CCW']

# ── helpers (same as generate_report_visualizations.py) ─────────────────────
class SimpleCfg:
    class data:
        append = 60; step = 2; stride = 1; classes = 5
        label_percentage = 0.5; leave_subject_out = 2
        apply_filter = False; filter_type = 'ema'; filter_alpha = 0.3
        dtw = False; single_test = False; share_train = 0.0
        kfold = None; force_num_subjects_train = None
        path = 'data/dataset'; train_files = []; test_files = []
        weighted_sampling = False
    class model:
        type = 'neuralnet'; num_fc_layers = 2
    class training:
        weighted_sampling = False
    class system:
        gpu = 0

def load_baseline_model(path):
    cfg = SimpleCfg()
    model = Conv1DNet(cfg)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model

def load_pruned_model(path):
    cfg = SimpleCfg()
    state = torch.load(path, map_location='cpu')
    fc1_out = state['fc_layers.0.weight'].shape[0]
    fc1_in  = state['fc_layers.0.weight'].shape[1]
    model = Conv1DNet(cfg)
    model.fc_layers = nn.Sequential(
        nn.Linear(fc1_in, fc1_out), nn.BatchNorm1d(fc1_out), nn.ReLU(),
        nn.Linear(fc1_out, cfg.data.classes),
    )
    model.load_state_dict(state)
    model.eval()
    return model

def apply_ema(data, alpha=0.3):
    out = np.zeros_like(data)
    out[0] = data[0]
    for i in range(1, len(data)):
        out[i] = alpha * data[i] + (1 - alpha) * out[i-1]
    return out

def make_windows(df, append=60, step=2):
    cols = ['acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z']
    data = df[cols].values.astype(np.float32)
    data[:, :3] /= 10.0;  data[:, 3:] /= 2.0
    labels = df['label'].values
    inputs, label_idx, center_idx = [], [], []
    n = len(data)
    for i in range(append * step - 1, n, step):
        idx = list(range(i - (append-1)*step, i+1, step))
        if len(idx) != append: continue
        window = data[idx]
        wlabels = labels[idx]
        lmax = wlabels.max()
        lbl = lmax if (lmax > 0 and (wlabels == lmax).sum() / append >= 0.5) else 0
        inputs.append(window)
        label_idx.append(lbl)
        center_idx.append(i)            # raw-data index of window center
    return np.stack(inputs), np.array(label_idx, dtype=np.int64), np.array(center_idx)

def run_inference(model, inputs, batch=512):
    all_preds = []
    with torch.no_grad():
        for s in range(0, len(inputs), batch):
            x = torch.tensor(inputs[s:s+batch]).permute(0, 2, 1).float()
            all_preds.append(model(x).argmax(1).cpu().numpy())
    return np.concatenate(all_preds)

# ── load ─────────────────────────────────────────────────────────────────────
print("Loading Subject 2 data...")
dfs = []
for seat in ['seating', 'standing']:
    f = f'data/dataset/ID02_{seat}_all_gestures.h5'
    if os.path.exists(f):
        dfs.append(pd.read_hdf(f, key='df'))
df_s02 = pd.concat(dfs).reset_index(drop=True)

raw_data = df_s02[['acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z']].values.astype(np.float32)
ema_data = apply_ema(raw_data, 0.3)
df_ema   = df_s02.copy()
df_ema[['acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z']] = ema_data

print("Building windows...")
inputs_raw, labels, centers = make_windows(df_s02)
inputs_ema, _,       _      = make_windows(df_ema)
print(f"  {len(labels)} windows, class dist: {np.bincount(labels)}")

print("Loading models...")
baseline_model  = load_baseline_model(
    'outputs_organized/05_archived_old_tests/filter_loo_baseline_s02_baseline/model_weights_10.pt')
optimized_model = load_pruned_model(
    'outputs/pruning/ema_s02_prune40pct_seed42/pruned_final.pt')

print("Running inference...")
preds_base = run_inference(baseline_model, inputs_raw)
preds_opt  = run_inference(optimized_model, inputs_ema)

# ── find contiguous gesture runs in the window-label array ───────────────────
# A "gesture run" = consecutive windows with the same non-zero label
print("\nScanning gesture runs...")

candidates = []
i = 0
n = len(labels)
while i < n:
    if labels[i] == 0:
        i += 1
        continue
    # start of a gesture run
    lbl = labels[i]
    j = i
    while j < n and labels[j] == lbl:
        j += 1
    # run is [i, j)
    run_len = j - i
    if run_len >= 3:   # at least 3 windows for meaningful stats
        seg_base = preds_base[i:j]
        seg_opt  = preds_opt[i:j]

        rest_pct    = (seg_base == 0).mean()
        correct_pct = (seg_opt  == lbl).mean()
        wrong_pct   = ((seg_opt > 0) & (seg_opt != lbl)).mean()

        # baseline: also count how many it got right / wrong class
        base_correct_pct = (seg_base == lbl).mean()
        base_wrong_pct   = ((seg_base > 0) & (seg_base != lbl)).mean()

        candidates.append({
            'win_start': i, 'win_end': j, 'label': lbl,
            'run_len': run_len,
            'rest_pct': rest_pct,
            'correct_pct': correct_pct,
            'wrong_pct': wrong_pct,
            'base_correct_pct': base_correct_pct,
            'base_wrong_pct': base_wrong_pct,
            'center_raw': centers[(i + j) // 2],
        })
    i = j

print(f"  Found {len(candidates)} gesture runs total.")

# ── filter by criteria ───────────────────────────────────────────────────────
good = [c for c in candidates if c['rest_pct'] >= 0.60 and c['correct_pct'] >= 0.60]
print(f"  {len(good)} runs meet rest_pct>=60% AND correct_pct>=60%.")

# Sort by how strong the contrast is
good.sort(key=lambda c: c['rest_pct'] + c['correct_pct'], reverse=True)

print("\n=== TOP CANDIDATES ===")
print(f"{'#':>3}  {'Label':10}  {'WinLen':>6}  "
      f"{'Base→Rest':>9}  {'Base→Correct':>12}  {'Base→Wrong':>10}  "
      f"{'EMA→Correct':>11}  {'EMA→Wrong':>9}  {'CenterRaw':>9}")
for k, c in enumerate(good[:20]):
    print(f"{k:3d}  {CLASS_NAMES[c['label']]:10}  {c['run_len']:6d}  "
          f"{c['rest_pct']*100:8.1f}%  {c['base_correct_pct']*100:11.1f}%  "
          f"{c['base_wrong_pct']*100:9.1f}%  "
          f"{c['correct_pct']*100:10.1f}%  {c['wrong_pct']*100:8.1f}%  "
          f"{c['center_raw']:9d}")

if good:
    best = good[0]
    print(f"\nBEST: Label={CLASS_NAMES[best['label']]}, "
          f"win_start={best['win_start']}, win_end={best['win_end']}, "
          f"center_raw={best['center_raw']}")
    print(f"  Baseline: {best['rest_pct']*100:.0f}% Rest, "
          f"{best['base_correct_pct']*100:.0f}% correct, "
          f"{best['base_wrong_pct']*100:.0f}% wrong class")
    print(f"  EMA:      {best['correct_pct']*100:.0f}% correct, "
          f"{best['wrong_pct']*100:.0f}% wrong class")
