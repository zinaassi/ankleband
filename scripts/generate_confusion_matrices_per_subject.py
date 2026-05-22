#!/usr/bin/env python3
"""
Individual confusion matrix figures for each test subject (S02, S03, S06).
Each figure: Baseline vs Optimised side-by-side, styled like the reference example.
Output: confusion_matrix_s0X.png  (300 DPI)
"""

import os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trainer.models.conv1d_model import Conv1DNet
import tables

OUT_DIR = "outputs/report_visualizations"
os.makedirs(OUT_DIR, exist_ok=True)

CLASS_NAMES = ['Rest', 'Grasp', 'Pinch', 'Rotate CW', 'Rotate CCW']

# ── helpers ───────────────────────────────────────────────────────────────────
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
        nn.Linear(fo, 5))
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

def build_cm(y_true, y_pred, n=5):
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def avg_recall(cm):
    """Mean per-class recall over gesture classes only (exclude Rest)."""
    per_class = np.diag(cm).astype(float) / cm.sum(axis=1).clip(min=1)
    return per_class[1:].mean()

def draw_cm(ax, cm, title):
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1).astype(float)
    cm_norm  = cm / row_sums

    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1,
                   interpolation='nearest', aspect='equal')

    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels(CLASS_NAMES, rotation=40, ha='right', fontsize=10)
    ax.set_yticklabels(CLASS_NAMES, fontsize=10)

    # cell annotations: bold count top, percentage below
    for r in range(5):
        for c in range(5):
            val  = cm[r, c]
            norm = cm_norm[r, c]
            fg   = 'white' if norm > 0.55 else 'black'
            is_notable = (r != c and norm > 0.10)
            txt_col = '#C62828' if (is_notable and fg == 'black') else fg
            weight  = 'bold'   if is_notable else 'normal'
            # top line: count (bold)
            ax.text(c, r - 0.18, f'{val:,}',
                    ha='center', va='center', fontsize=9,
                    color=fg, fontweight='bold')
            # bottom line: percentage
            ax.text(c, r + 0.22, f'({norm:.1%})',
                    ha='center', va='center', fontsize=8.5,
                    color=txt_col, fontweight=weight)

    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    ax.set_ylabel('True label', fontsize=10)
    return im

# ── subject configs ───────────────────────────────────────────────────────────
SUBJECTS = {
    2: dict(
        label='Subject 2 (LOSO)',
        note='median performer',
        baseline='outputs_organized/05_archived_old_tests/'
                 'filter_loo_baseline_s02_baseline/model_weights_10.pt',
        pruned='outputs/pruning/ema_s02_prune40pct_seed42/pruned_final.pt',
    ),
    3: dict(
        label='Subject 3 (LOSO)',
        note='best overall recall',
        baseline='outputs_organized/05_archived_old_tests/'
                 'filter_loo_baseline_s03_baseline/model_weights_10.pt',
        pruned='outputs/pruning/ema_s03_prune40pct_seed42/pruned_final.pt',
    ),
    6: dict(
        label='Subject 6 (LOSO)',
        note='largest drop after pruning',
        baseline='outputs_organized/05_archived_old_tests/'
                 'filter_loo_baseline_s06_baseline/model_weights_10.pt',
        pruned='outputs/pruning/ema_s06_prune40pct_seed42/pruned_final.pt',
    ),
}

def load_subject(sid):
    dfs = []
    for tag in ['seating', 'standing']:
        f = f'data/dataset/ID{sid:02d}_{tag}_all_gestures.h5'
        if os.path.exists(f):
            dfs.append(pd.read_hdf(f, key='df'))
    return pd.concat(dfs).reset_index(drop=True)

# ── generate one figure per subject ───────────────────────────────────────────
for sid, cfg in SUBJECTS.items():
    print(f"Processing {cfg['label']}...")
    df      = load_subject(sid)
    X_raw, y = make_windows(df, use_ema=False)
    X_ema, _ = make_windows(df, use_ema=True)

    bm = load_baseline(cfg['baseline'])
    om = load_pruned(cfg['pruned'])

    pb = run_inference(bm, X_raw)
    po = run_inference(om, X_ema)

    cm_b = build_cm(y, pb)
    cm_o = build_cm(y, po)

    rec_b = avg_recall(cm_b)
    rec_o = avg_recall(cm_o)
    delta = (rec_o - rec_b) * 100
    print(f"  Baseline avg recall: {rec_b:.3f}  |  Optimised: {rec_o:.3f}  |  delta: {delta:+.2f} pp")

    # ── figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), facecolor='white')
    plt.subplots_adjust(wspace=0.35)

    im_b = draw_cm(axes[0], cm_b,
                   title='Baseline (FP32, No Filter)')
    im_o = draw_cm(axes[1], cm_o,
                   title='Optimised (Pruned FP32 + EMA \u03b1=0.3)')

    # x-axis labels with avg recall
    axes[0].set_xlabel(f'Predicted label\nAvg Recall: {rec_b:.1%}', fontsize=10)
    axes[1].set_xlabel(f'Predicted label\nAvg Recall: {rec_o:.1%}  ({delta:+.2f} pp)',
                       fontsize=10)

    # shared colorbar on the right
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cb = fig.colorbar(im_o, cax=cbar_ax)
    cb.set_label('Normalised count\n(per true class)', fontsize=9)
    cb.ax.tick_params(labelsize=8)

    fig.suptitle(
        f'Confusion Matrix: Baseline vs. Optimized Model \u2014 {cfg["label"]}',
        fontsize=13, fontweight='bold', y=1.02)

    out_path = f'{OUT_DIR}/confusion_matrix_s{sid:02d}.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved -> {out_path}")

print("\nDone.")
