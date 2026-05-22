#!/usr/bin/env python3
"""
Confusion matrices for all 3 test subjects (S02, S03, S06):
  Baseline (no filter) vs Optimised (EMA α=0.3 + 40% pruning)
Saved as outputs/report_visualizations/confusion_matrix_all_subjects.png
"""

import os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trainer.models.conv1d_model import Conv1DNet
import tables

OUT_DIR = "outputs/report_visualizations"
os.makedirs(OUT_DIR, exist_ok=True)

CLASS_NAMES = ['Rest', 'Grasp', 'Pinch', 'Rot. CW', 'Rot. CCW']

# ── helpers ──────────────────────────────────────────────────────────────────
class Cfg:
    class data:
        append=60; step=2; stride=1; classes=5; label_percentage=0.5
        leave_subject_out=2; apply_filter=False; filter_type='ema'
        filter_alpha=0.3; dtw=False; single_test=False; share_train=0.0
        kfold=None; force_num_subjects_train=None; path='data/dataset'
        train_files=[]; test_files=[]; weighted_sampling=False
    class model: type='neuralnet'; num_fc_layers=2
    class training: weighted_sampling=False
    class system: gpu=0

def load_baseline(path):
    m = Conv1DNet(Cfg())
    m.load_state_dict(torch.load(path, map_location='cpu'))
    m.eval(); return m

def load_pruned(path):
    s = torch.load(path, map_location='cpu')
    fo = s['fc_layers.0.weight'].shape[0]
    fi = s['fc_layers.0.weight'].shape[1]
    m = Conv1DNet(Cfg())
    m.fc_layers = nn.Sequential(
        nn.Linear(fi, fo), nn.BatchNorm1d(fo), nn.ReLU(),
        nn.Linear(fo, Cfg.data.classes))
    m.load_state_dict(s); m.eval(); return m

def apply_ema(data, alpha=0.3):
    out = np.zeros_like(data); out[0] = data[0]
    for i in range(1, len(data)):
        out[i] = alpha * data[i] + (1 - alpha) * out[i-1]
    return out

def make_windows(df, use_ema=False, append=60, step=2):
    cols = ['acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z']
    d = df[cols].values.astype(np.float32)
    if use_ema:
        d = apply_ema(d)
    d[:, :3] /= 10.0; d[:, 3:] /= 2.0
    lbl = df['label'].values
    X, y = [], []
    for i in range(append * step - 1, len(d), step):
        idx = list(range(i - (append-1)*step, i+1, step))
        if len(idx) != append: continue
        wl = lbl[idx]; lmax = wl.max()
        l = lmax if (lmax > 0 and (wl == lmax).sum() / append >= 0.5) else 0
        X.append(d[idx]); y.append(l)
    return np.stack(X), np.array(y, dtype=np.int64)

def run_inference(model, X, batch=512):
    preds = []
    with torch.no_grad():
        for s in range(0, len(X), batch):
            x = torch.tensor(X[s:s+batch]).permute(0, 2, 1).float()
            preds.append(model(x).argmax(1).cpu().numpy())
    return np.concatenate(preds)

def confusion_matrix(y_true, y_pred, n=5):
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def recall_from_cm(cm):
    diag = np.diag(cm).astype(float)
    row  = cm.sum(axis=1).clip(min=1).astype(float)
    # exclude rest (class 0) from gesture recall
    return diag[1:].sum() / row[1:].sum()

# ── subjects ─────────────────────────────────────────────────────────────────
SUBJECTS = {
    2: {
        'baseline': 'outputs_organized/05_archived_old_tests/'
                    'filter_loo_baseline_s02_baseline/model_weights_10.pt',
        'pruned':   'outputs/pruning/ema_s02_prune40pct_seed42/pruned_final.pt',
        'data_ids': [2],
    },
    3: {
        'baseline': 'outputs_organized/05_archived_old_tests/'
                    'filter_loo_baseline_s03_baseline/model_weights_10.pt',
        'pruned':   'outputs/pruning/ema_s03_prune40pct_seed42/pruned_final.pt',
        'data_ids': [3],
    },
    6: {
        'baseline': 'outputs_organized/05_archived_old_tests/'
                    'filter_loo_baseline_s06_baseline/model_weights_10.pt',
        'pruned':   'outputs/pruning/ema_s06_prune40pct_seed42/pruned_final.pt',
        'data_ids': [6],
    },
}

def load_subject_data(subject_id):
    dfs = []
    for tag in ['seating', 'standing']:
        f = f'data/dataset/ID{subject_id:02d}_{tag}_all_gestures.h5'
        if os.path.exists(f):
            dfs.append(pd.read_hdf(f, key='df'))
    return pd.concat(dfs).reset_index(drop=True)

# ── build all confusion matrices ──────────────────────────────────────────────
results = {}
for sid, cfg in SUBJECTS.items():
    print(f"Processing Subject {sid}...")
    df = load_subject_data(sid)
    X_raw, y = make_windows(df, use_ema=False)
    X_ema, _ = make_windows(df, use_ema=True)

    bm = load_baseline(cfg['baseline'])
    om = load_pruned(cfg['pruned'])

    pb = run_inference(bm, X_raw)
    po = run_inference(om, X_ema)

    cm_b = confusion_matrix(y, pb)
    cm_o = confusion_matrix(y, po)

    rec_b = recall_from_cm(cm_b)
    rec_o = recall_from_cm(cm_o)
    print(f"  Baseline recall: {rec_b:.4f}  |  Optimised recall: {rec_o:.4f}  "
          f"| delta: {(rec_o-rec_b)*100:+.2f} pp")

    results[sid] = dict(cm_b=cm_b, cm_o=cm_o, rec_b=rec_b, rec_o=rec_o)

# ── plot: 3 rows (subjects) × 2 cols (baseline / optimised) ──────────────────
fig, axes = plt.subplots(3, 2, figsize=(12, 16), facecolor='white')
plt.subplots_adjust(hspace=0.55, wspace=0.25)

def draw_cm(ax, cm, line1, line2):
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1).astype(float)
    cm_norm  = cm / row_sums

    ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1, interpolation='nearest')

    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels(CLASS_NAMES, rotation=40, ha='right', fontsize=8)
    ax.set_yticklabels(CLASS_NAMES, fontsize=8)
    ax.set_xlabel('Predicted', fontsize=8, labelpad=2)
    ax.set_ylabel('True', fontsize=8)
    # two-line title: bold first line, smaller second line via padding trick
    ax.set_title(f'{line1}\n{line2}', fontsize=9, fontweight='bold',
                 linespacing=1.5, pad=6)

    # annotate cells — show % always, count only when meaningful
    for r in range(5):
        for c in range(5):
            val  = cm[r, c]
            norm = cm_norm[r, c]
            if norm < 0.005 and val == 0:
                continue           # skip true zeros entirely
            fg = 'white' if norm > 0.55 else 'black'
            is_notable_error = (r != c and norm > 0.10)
            txt_color = '#C62828' if (is_notable_error and fg == 'black') else fg
            weight    = 'bold'   if is_notable_error else 'normal'
            # single line: "82% (1234)" — no newline, smaller font
            label = f'{norm:.0%} ({val})' if norm >= 0.02 else f'{norm:.0%}'
            ax.text(c, r, label, ha='center', va='center',
                    fontsize=6.5, color=txt_color, fontweight=weight)

SUBJ_LABELS = {2: 'Subject 2', 3: 'Subject 3', 6: 'Subject 6'}
SUBJ_NOTES  = {2: 'median performer',
               3: 'best overall recall',
               6: 'largest drop after pruning'}

for row, sid in enumerate([2, 3, 6]):
    r   = results[sid]
    lbl = SUBJ_LABELS[sid]
    note = SUBJ_NOTES[sid]
    delta = (r['rec_o'] - r['rec_b']) * 100

    draw_cm(axes[row, 0], r['cm_b'],
            line1=f'Baseline \u2014 {lbl} ({note})',
            line2=f'Gesture recall: {r["rec_b"]:.3f}')

    draw_cm(axes[row, 1], r['cm_o'],
            line1=f'Optimised \u2014 {lbl} ({note})',
            line2=f'Gesture recall: {r["rec_o"]:.3f}   ({delta:+.2f} pp vs baseline)')

fig.suptitle('Confusion Matrices: Baseline vs. Optimised (EMA \u03b1=0.3 + 40% Pruning)\n'
             'All Three Test Subjects  \u2014  bold red = off-diagonal confusion > 10%',
             fontsize=12, fontweight='bold', y=1.01)

out_path = f'{OUT_DIR}/confusion_matrix_all_subjects.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\nSaved -> {out_path}")
