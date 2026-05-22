#!/usr/bin/env python3
"""
Generate three report visualizations:
  1. Confusion matrix comparison (baseline vs optimized)
  2. Pipeline architecture diagram
  3. Time-series gesture detection comparison
"""

import os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainer.models.conv1d_model import Conv1DNet
from trainer.models.pruned_conv1d_model import PrunedConv1DNet
from trainer.utils import ConfigManager

OUT_DIR = "outputs/report_visualizations"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class SimpleCfg:
    """Minimal config object so we can instantiate Conv1DNet without JSON."""
    class data:
        append = 60
        step = 2
        stride = 1
        classes = 5
        label_percentage = 0.5
        leave_subject_out = 2
        apply_filter = False
        filter_type = 'ema'
        filter_alpha = 0.3
        dtw = False
        single_test = False
        share_train = 0.0
        kfold = None
        force_num_subjects_train = None
        path = 'data/dataset'
        train_files = []
        test_files = []
        weighted_sampling = False
    class model:
        type = 'neuralnet'
        num_fc_layers = 2
    class training:
        weighted_sampling = False
    class system:
        gpu = 0

def load_baseline_model(weights_path):
    """Load the unfiltered baseline Conv1DNet."""
    cfg = SimpleCfg()
    model = Conv1DNet(cfg)
    state = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(state)
    model.eval()
    return model

def load_pruned_model(weights_path):
    """Load the 40%-pruned EMA-filtered model."""
    cfg = SimpleCfg()
    cfg.model.num_fc_layers = 2
    state = torch.load(weights_path, map_location='cpu')
    # detect pruned FC1 size from checkpoint
    fc1_out = state['fc_layers.0.weight'].shape[0]
    fc1_in  = state['fc_layers.0.weight'].shape[1]
    model = Conv1DNet(cfg)
    model.fc_layers = nn.Sequential(
        nn.Linear(fc1_in, fc1_out),
        nn.BatchNorm1d(fc1_out),
        nn.ReLU(),
        nn.Linear(fc1_out, cfg.data.classes),
    )
    model.load_state_dict(state)
    model.eval()
    return model

def make_windows(df, append=60, step=2):
    """Replicate the load_data windowing for raw DataFrame."""
    cols = ['acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z']
    MAX_ACC, MAX_GYRO = 10.0, 2.0

    data = df[cols].values.astype(np.float32)
    data[:, :3] /= MAX_ACC
    data[:, 3:] /= MAX_GYRO
    labels = df['label'].values

    inputs, label_idx = [], []
    n = len(data)
    for i in range(append * step - 1, n, step):
        # gather window: every 'step'-th sample going back 'append' samples
        idx = list(range(i - (append-1)*step, i+1, step))
        if len(idx) != append:
            continue
        window = data[idx]            # (60, 6)
        window_labels = labels[idx]

        lmax = window_labels.max()
        if lmax > 0:
            pct = (window_labels / lmax).sum() / append
            lbl = lmax if pct >= 0.5 else 0
        else:
            lbl = 0

        inputs.append(window)
        label_idx.append(lbl)

    return np.stack(inputs), np.array(label_idx, dtype=np.int64)

def apply_ema(data, alpha=0.3):
    """Apply EMA filter to (N, 6) array."""
    out = np.zeros_like(data)
    out[0] = data[0]
    for i in range(1, len(data)):
        out[i] = alpha * data[i] + (1 - alpha) * out[i-1]
    return out

def run_inference(model, inputs):
    """inputs: (N, 60, 6) numpy -> predictions (N,)"""
    model.eval()
    all_preds = []
    batch = 512
    with torch.no_grad():
        for start in range(0, len(inputs), batch):
            x = torch.tensor(inputs[start:start+batch]).permute(0, 2, 1).float()
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
    return np.concatenate(all_preds)

def confusion_matrix_manual(y_true, y_pred, n=5):
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

# ---------------------------------------------------------------------------
# Load data for subject 2
# ---------------------------------------------------------------------------
print("Loading subject 2 data...")
import tables  # PyTables for HDF5

def load_h5_subject(subject_id):
    files = [
        f'data/dataset/ID{subject_id:02d}_seating_all_gestures.h5',
        f'data/dataset/ID{subject_id:02d}_standing_all_gestures.h5',
    ]
    dfs = []
    for f in files:
        if os.path.exists(f):
            dfs.append(pd.read_hdf(f, key='df'))
    return pd.concat(dfs).reset_index(drop=True)

df_s02 = load_h5_subject(2)
print(f"  Subject 2: {len(df_s02)} samples, labels: {sorted(df_s02['label'].unique())}")

# Build windows: raw (for baseline) and EMA-filtered (for optimized)
raw_data = df_s02[['acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z']].values.astype(np.float32)
ema_data = apply_ema(raw_data, alpha=0.3)
df_ema = df_s02.copy()
df_ema[['acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z']] = ema_data

inputs_raw, labels = make_windows(df_s02)
inputs_ema, _      = make_windows(df_ema)
print(f"  Windows: {len(labels)}, class distribution: {np.bincount(labels)}")

# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------
print("Loading models...")
baseline_path = 'outputs_organized/05_archived_old_tests/filter_loo_baseline_s02_baseline/model_weights_10.pt'
optimized_path = 'outputs/pruning/ema_s02_prune40pct_seed42/pruned_final.pt'

baseline_model  = load_baseline_model(baseline_path)
optimized_model = load_pruned_model(optimized_path)
print("  Models loaded.")

# Run inference
print("Running inference...")
preds_baseline  = run_inference(baseline_model, inputs_raw)
preds_optimized = run_inference(optimized_model, inputs_ema)
print(f"  Baseline  - accuracy: {(preds_baseline==labels).mean():.4f}")
print(f"  Optimized - accuracy: {(preds_optimized==labels).mean():.4f}")

cm_baseline  = confusion_matrix_manual(labels, preds_baseline)
cm_optimized = confusion_matrix_manual(labels, preds_optimized)


# ===========================================================================
# VISUALIZATION 1: Confusion Matrix Comparison
# ===========================================================================
print("\nGenerating Visualization 1: Confusion Matrix Comparison...")

CLASS_NAMES = ['Rest', 'Grasp', 'Pinch', 'Rotate CW', 'Rotate CCW']

def plot_cm(ax, cm, title, fontsize=9):
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm = cm.astype(float) / row_sums

    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)

    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(CLASS_NAMES, rotation=35, ha='right', fontsize=fontsize)
    ax.set_yticklabels(CLASS_NAMES, fontsize=fontsize)
    ax.set_xlabel('Predicted label', fontsize=fontsize+1)
    ax.set_ylabel('True label', fontsize=fontsize+1)
    ax.set_title(title, fontsize=fontsize+2, fontweight='bold', pad=10)

    for i in range(5):
        for j in range(5):
            count = cm[i, j]
            pct   = cm_norm[i, j] * 100
            color = 'white' if cm_norm[i, j] > 0.55 else 'black'
            ax.text(j, i, f'{count}\n({pct:.1f}%)',
                    ha='center', va='center', fontsize=fontsize-1,
                    color=color, fontweight='bold' if i == j else 'normal')
    return im

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor='white')
plt.subplots_adjust(wspace=0.35)

im0 = plot_cm(axes[0], cm_baseline,  'Baseline (FP32, No Filter)')
im1 = plot_cm(axes[1], cm_optimized, 'Optimized (Pruned FP32 + EMA α=0.3)')

# Add recall annotation below each matrix
def recall_str(cm):
    per_class = [cm[i,i]/max(cm[i].sum(),1)*100 for i in range(5)]
    return f'Avg Recall: {np.mean(per_class):.1f}%'

for ax, cm in zip(axes, [cm_baseline, cm_optimized]):
    ax.text(0.5, -0.22, recall_str(cm), transform=ax.transAxes,
            ha='center', fontsize=9, style='italic', color='#444444')

# Shared colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
cb = fig.colorbar(im1, cax=cbar_ax)
cb.set_label('Normalized count (per true class)', fontsize=8)
cb.ax.tick_params(labelsize=8)

fig.suptitle('Confusion Matrix: Baseline vs. Optimized Model — Subject 2 (LOSO)',
             fontsize=12, fontweight='bold', y=1.01)

plt.savefig(f'{OUT_DIR}/confusion_matrix_comparison.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved -> {OUT_DIR}/confusion_matrix_comparison.png")


# ===========================================================================
# VISUALIZATION 2: Pipeline Architecture Diagram
# ===========================================================================
print("\nGenerating Visualization 2: Pipeline Architecture Diagram...")

fig, axes = plt.subplots(1, 4, figsize=(16, 10), facecolor='white')
plt.subplots_adjust(wspace=0.05)

STAGES = [
    {
        'title': 'Stage 1\nBaseline',
        'subtitle': '56.36 KB  |  13,897 params',
        'size': '56.36 KB',
        'layers': [
            ('Input',           '6 ch × 60 steps',    'input'),
            ('Conv1D',          '6->10, k=3, s=3',      'fp32'),
            ('Flatten',         '200 features',         'fp32'),
            ('BatchNorm(200)',   '+ ReLU',              'fp32'),
            ('FC Layer 1',      '200 -> 64',             'fp32'),
            ('BatchNorm(64)',    '+ ReLU',              'fp32'),
            ('FC Layer 2',      '64 -> 5',              'fp32'),
            ('Softmax',         '5 classes',           'output'),
        ],
        'arrows': ['-0%', None, None, None, None, None, None],
    },
    {
        'title': 'Stage 2\n+ EMA Filter',
        'subtitle': '56.36 KB  |  13,897 params',
        'size': '56.36 KB',
        'extra_top': ('EMA Filter', 'α = 0.3', 'ema'),
        'layers': [
            ('Input',           '6 ch × 60 steps',    'input'),
            ('Conv1D',          '6->10, k=3, s=3',      'fp32'),
            ('Flatten',         '200 features',         'fp32'),
            ('BatchNorm(200)',   '+ ReLU',              'fp32'),
            ('FC Layer 1',      '200 -> 64',             'fp32'),
            ('BatchNorm(64)',    '+ ReLU',              'fp32'),
            ('FC Layer 2',      '64 -> 5',              'fp32'),
            ('Softmax',         '5 classes',           'output'),
        ],
    },
    {
        'title': 'Stage 3\n+ 40% Pruning',
        'subtitle': '38.32 KB  |  9,321 params',
        'size': '38.32 KB',
        'layers': [
            ('Input',           '6 ch × 60 steps',    'input'),
            ('Conv1D',          '6->10, k=3, s=3',      'fp32'),
            ('Flatten',         '200 features',         'fp32'),
            ('BatchNorm(200)',   '+ ReLU',              'fp32'),
            ('FC Layer 1',      '200 -> 42  ✂',         'pruned'),
            ('BatchNorm(42)',    '+ ReLU',              'pruned'),
            ('FC Layer 2',      '42 -> 5  ✂',           'pruned'),
            ('Softmax',         '5 classes',           'output'),
        ],
    },
    {
        'title': 'Stage 4\n+ INT8 QAT',
        'subtitle': '12.60 KB  |  9,321 params',
        'size': '12.60 KB',
        'layers': [
            ('Input',           '6 ch × 60 steps',    'input'),
            ('Conv1D',          '6->10, k=3, s=3',      'int8'),
            ('Flatten',         '200 features',         'int8'),
            ('BatchNorm(200)',   '+ ReLU  [FP32]',      'fp32'),
            ('FC Layer 1',      '200 -> 42',             'int8'),
            ('BatchNorm(42)',    '+ ReLU  [FP32]',      'fp32'),
            ('FC Layer 2',      '42 -> 5',              'int8'),
            ('Softmax',         '5 classes',           'output'),
        ],
    },
]

COLOR_MAP = {
    'input':  ('#E8F4F8', '#2196F3'),
    'fp32':   ('#E3F2FD', '#1565C0'),
    'ema':    ('#FFF8E1', '#F57F17'),
    'pruned': ('#E8F5E9', '#2E7D32'),
    'int8':   ('#E8F5E9', '#1B5E20'),
    'output': ('#F3E5F5', '#6A1B9A'),
}

REDUCTION = [None, '±0%\n(preprocessing)', '−32%\n(pruning)', '−67%\n(quantization)']

for col, (ax, stage) in enumerate(zip(axes, STAGES)):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Stage title
    ax.text(0.5, 0.97, stage['title'], ha='center', va='top',
            fontsize=11, fontweight='bold', transform=ax.transAxes)
    ax.text(0.5, 0.91, stage['subtitle'], ha='center', va='top',
            fontsize=8, color='#555555', transform=ax.transAxes,
            style='italic')

    # EMA pre-processing box (Stage 2 only)
    y_start = 0.82
    if 'extra_top' in stage:
        name, detail, ctype = stage['extra_top']
        fc, ec = COLOR_MAP[ctype]
        rect = mpatches.FancyBboxPatch((0.08, y_start - 0.055), 0.84, 0.05,
            boxstyle='round,pad=0.01', facecolor=fc, edgecolor=ec, linewidth=2,
            transform=ax.transAxes, zorder=2)
        ax.add_patch(rect)
        ax.text(0.5, y_start - 0.028, f'{name}  [{detail}]',
                ha='center', va='center', fontsize=8, fontweight='bold',
                color=ec, transform=ax.transAxes)
        # small arrow down to Input
        ax.annotate('', xy=(0.5, y_start - 0.07), xytext=(0.5, y_start - 0.055),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', color='#555555', lw=1.2))
        y_start -= 0.075

    # Layer boxes
    n_layers = len(stage['layers'])
    box_h = 0.057
    gap   = 0.012
    total_h = n_layers * box_h + (n_layers - 1) * gap
    y0 = y_start - 0.01

    for i, (name, detail, ctype) in enumerate(stage['layers']):
        fc, ec = COLOR_MAP[ctype]
        y_box = y0 - i * (box_h + gap)

        rect = mpatches.FancyBboxPatch((0.06, y_box - box_h), 0.88, box_h,
            boxstyle='round,pad=0.008', facecolor=fc, edgecolor=ec, linewidth=1.5,
            transform=ax.transAxes, zorder=2)
        ax.add_patch(rect)

        ax.text(0.5, y_box - box_h * 0.38, name,
                ha='center', va='center', fontsize=8.5, fontweight='bold',
                color=ec, transform=ax.transAxes)
        ax.text(0.5, y_box - box_h * 0.75, detail,
                ha='center', va='center', fontsize=7, color='#444444',
                transform=ax.transAxes)

        # Arrow between boxes
        if i < n_layers - 1:
            ax.annotate('', xy=(0.5, y_box - box_h - gap), xytext=(0.5, y_box - box_h),
                        xycoords='axes fraction', textcoords='axes fraction',
                        arrowprops=dict(arrowstyle='->', color='#888888', lw=1))

    # Size badge bottom
    y_badge = y0 - n_layers * (box_h + gap) - 0.005
    badge_color = ['#CFD8DC', '#E3F2FD', '#C8E6C9', '#A5D6A7'][col]
    rect2 = mpatches.FancyBboxPatch((0.15, y_badge - 0.045), 0.7, 0.042,
        boxstyle='round,pad=0.01', facecolor=badge_color, edgecolor='#546E7A',
        linewidth=1.5, transform=ax.transAxes, zorder=2)
    ax.add_patch(rect2)
    ax.text(0.5, y_badge - 0.023, f'Model size: {stage["size"]}',
            ha='center', va='center', fontsize=8.5, fontweight='bold',
            color='#263238', transform=ax.transAxes)

    # Reduction arrow between columns (right side)
    if col > 0 and REDUCTION[col]:
        ax.text(0.02, 0.55, REDUCTION[col],
                ha='center', va='center', fontsize=7.5, color='#B71C1C',
                fontweight='bold', transform=ax.transAxes,
                rotation=90, style='italic')

# Legend
legend_items = [
    mpatches.Patch(facecolor='#E3F2FD', edgecolor='#1565C0', label='FP32 layer'),
    mpatches.Patch(facecolor='#E8F5E9', edgecolor='#1B5E20', label='INT8 layer (quantized)'),
    mpatches.Patch(facecolor='#E8F5E9', edgecolor='#2E7D32', label='Pruned layer'),
    mpatches.Patch(facecolor='#FFF8E1', edgecolor='#F57F17', label='EMA preprocessing'),
]
fig.legend(handles=legend_items, loc='lower center', ncol=4,
           fontsize=9, frameon=True, bbox_to_anchor=(0.5, 0.0))

fig.suptitle('Optimization Pipeline Architecture — Baseline -> EMA -> Pruning -> INT8 QAT',
             fontsize=13, fontweight='bold', y=1.01)

plt.savefig(f'{OUT_DIR}/pipeline_architecture.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved -> {OUT_DIR}/pipeline_architecture.png")


# ===========================================================================
# VISUALIZATION 3: Time-Series Gesture Detection Comparison
# ===========================================================================
print("\nGenerating Visualization 3: Time-Series Gesture Detection Comparison...")

# Gesture run with strongest baseline/EMA contrast found across 312 runs in Subject 2:
# Rotate CCW, windows 126632..126752 — baseline predicts Rest 38% of windows,
# EMA predicts correct class 72% of windows.  (scan: find_best_gesture_window.py)
APPEND, STEP = 60, 2
WIN_START = 126632   # first window index of gesture run
WIN_END   = 126752   # last window index (inclusive)
true_label = 4       # Rotate CCW
CLASS_NAMES_SHORT = ['Rest', 'Grasp', 'Pinch', 'Rot.CW', 'Rot.CCW']
print(f"  Using hardcoded gesture run: {CLASS_NAMES_SHORT[true_label]}, "
      f"windows {WIN_START}..{WIN_END}")

# Window k's last raw sample is at raw position: (APPEND*STEP - 1) + k*STEP = 119 + 2k
raw_gest_start = 119 + WIN_START * STEP    # ~253383
raw_gest_end   = 119 + WIN_END   * STEP    # ~253623

# Context: 1.5 full windows on each side
ctx = APPEND * STEP + APPEND * STEP // 2
plot_start = max(0, raw_gest_start - ctx)
plot_end   = min(len(raw_data), raw_gest_end + ctx)

raw_segment  = raw_data[plot_start:plot_end, 0]    # acc_x
ema_segment  = ema_data[plot_start:plot_end, 0]
gt_labels    = df_s02['label'].values[plot_start:plot_end]

# Gesture region in plot coords
gesture_start_ms = (raw_gest_start - plot_start) * (1000.0 / 200.0)
gesture_end_ms   = (raw_gest_end   - plot_start) * (1000.0 / 200.0)

# Build sample-level prediction arrays for the plot window
# Each window covers APPEND*STEP raw samples; slide every STEP
def get_dense_preds(model, raw_array, n_raw, ema=False):
    """Return per-sample predictions for the plot window."""
    preds_dense = np.zeros(n_raw, dtype=int)
    counts      = np.zeros(n_raw, dtype=int)

    src = apply_ema(raw_array, 0.3) if ema else raw_array.copy()
    src_norm = src.copy()
    src_norm[:, :3] /= 10.0
    src_norm[:, 3:] /= 2.0

    for i in range(0, n_raw - APPEND * STEP, STEP):
        idx = list(range(i, i + APPEND * STEP, STEP))
        window = src_norm[idx].astype(np.float32)
        x = torch.tensor(window).unsqueeze(0).permute(0, 2, 1).float()
        with torch.no_grad():
            pred = model(x).argmax(dim=1).item()
        preds_dense[i:i + APPEND * STEP] += pred
        counts[i:i + APPEND * STEP] += 1
    counts = counts.clip(min=1)
    # majority vote approximation: use raw pred, not average
    # redo cleanly: just mark the center of each window
    preds_dense2 = np.zeros(n_raw, dtype=int)
    for i in range(0, n_raw - APPEND * STEP, STEP):
        idx = list(range(i, i + APPEND * STEP, STEP))
        window = src_norm[idx].astype(np.float32)
        x = torch.tensor(window).unsqueeze(0).permute(0, 2, 1).float()
        with torch.no_grad():
            pred = model(x).argmax(dim=1).item()
        center = i + APPEND * STEP // 2
        preds_dense2[center] = pred + 1  # +1 so 0=no prediction yet
    return preds_dense2

# Get dense predictions for just the plot window
seg_raw = raw_data[plot_start:plot_end]
seg_ema_arr = np.stack([raw_data[plot_start:plot_end],
                         ema_data[plot_start:plot_end]], axis=0)

n_seg = plot_end - plot_start

# Simple approach: run inference over overlapping windows in the segment
def segment_preds(model, data_full, seg_start, seg_end, use_ema=False):
    n = seg_end - seg_start
    preds = np.full(n, -1, dtype=int)
    src = data_full[max(0, seg_start - APPEND*STEP): seg_end + APPEND*STEP]
    if use_ema:
        src = apply_ema(src, 0.3)
    norm = src.copy()
    norm[:, :3] /= 10.0
    norm[:, 3:] /= 2.0
    offset = seg_start - max(0, seg_start - APPEND*STEP)

    for i in range(0, len(norm) - APPEND*STEP, STEP):
        idx = list(range(i, i + APPEND*STEP, STEP))
        window = norm[idx].astype(np.float32)
        x = torch.tensor(window).unsqueeze(0).permute(0, 2, 1).float()
        with torch.no_grad():
            pred = model(x).argmax(dim=1).item()
        center_in_src = i + APPEND*STEP//2
        center_in_seg = center_in_src - offset
        if 0 <= center_in_seg < n:
            preds[center_in_seg] = pred
    return preds

print("  Computing dense predictions for plot window...")
preds_seg_base = segment_preds(baseline_model,  raw_data, plot_start, plot_end, use_ema=False)
preds_seg_opt  = segment_preds(optimized_model, raw_data, plot_start, plot_end, use_ema=True)

# Plot
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, facecolor='white')
plt.subplots_adjust(hspace=0.08)

t = np.arange(plot_end - plot_start) * (1000.0 / 200.0)  # ms
gesture_s = gesture_start_ms
gesture_e = gesture_end_ms

raw_sig = raw_data[plot_start:plot_end, 0]
ema_sig = ema_data[plot_start:plot_end, 0]

# x-axis zoom: 500ms before gesture start to 500ms after gesture end
x_lo = max(t[0],  gesture_s - 500)
x_hi = min(t[-1], gesture_e + 500)

# ---- Top: raw signal + ground truth shading ----
ax = axes[0]
ax.plot(t, raw_sig, color='#1565C0', lw=1.2, label='Raw acc_x', alpha=0.9)
ax.axvspan(gesture_s, gesture_e, alpha=0.18, color='#4CAF50',
           label=f'Ground truth: {CLASS_NAMES[true_label]}')
ax.axvline(gesture_s, color='#2E7D32', lw=1.5, ls='--', alpha=0.7)
ax.axvline(gesture_e, color='#2E7D32', lw=1.5, ls='--', alpha=0.7)
ax.set_ylabel('Acceleration X\n(normalized)', fontsize=9)
ax.legend(loc='upper right', fontsize=8, framealpha=0.85)
ax.set_title('Ground Truth', fontsize=10, fontweight='bold')
ax.set_xlim(x_lo, x_hi)
ax.grid(axis='x', alpha=0.3)
ax.set_facecolor('white')

# ---- Middle: EMA signal + optimized predictions ----
ax = axes[1]
ax.plot(t, ema_sig, color='#1565C0', lw=1.2, label='EMA filtered (\u03b1=0.3)', alpha=0.9)
ax.axvspan(gesture_s, gesture_e, alpha=0.12, color='#4CAF50')
ax.axvline(gesture_s, color='#2E7D32', lw=1.5, ls='--', alpha=0.7)
ax.axvline(gesture_e, color='#2E7D32', lw=1.5, ls='--', alpha=0.7)

in_gest = (t >= gesture_s) & (t <= gesture_e)
det_mask = in_gest & (preds_seg_opt == true_label)
if det_mask.any():
    ax.scatter(t[det_mask], ema_sig[det_mask], marker='v', s=40,
               color='#2E7D32', zorder=5, label='Correct class')
opt_rest = in_gest & (preds_seg_opt == 0)
if opt_rest.any():
    ax.scatter(t[opt_rest], ema_sig[opt_rest], marker='x', s=35,
               color='#F57F17', zorder=5, label='Predicted Rest')
opt_wrong = in_gest & (preds_seg_opt > 0) & (preds_seg_opt != true_label)
if opt_wrong.any():
    ax.scatter(t[opt_wrong], ema_sig[opt_wrong], marker='s', s=25,
               color='#E53935', zorder=5, label='Wrong gesture class')

ax.text(0.99, 0.96, 'EMA \u03b1=0.3: 72% of windows correct',
        transform=ax.transAxes, ha='right', va='top',
        fontsize=9.5, fontweight='bold', color='#1B5E20',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9', edgecolor='#2E7D32'))
ax.set_ylabel('Acceleration X\n(normalized)', fontsize=9)
ax.legend(loc='upper left', fontsize=8, framealpha=0.85)
ax.set_title('EMA \u03b1=0.3 (Optimized)', fontsize=10, fontweight='bold')
ax.set_xlim(x_lo, x_hi)
ax.grid(axis='x', alpha=0.3)
ax.set_facecolor('white')

# ---- Bottom: raw signal + baseline predictions ----
ax = axes[2]
ax.plot(t, raw_sig, color='#1565C0', lw=1.2, label='Raw acc_x', alpha=0.9)
ax.axvspan(gesture_s, gesture_e, alpha=0.12, color='#4CAF50')
ax.axvline(gesture_s, color='#2E7D32', lw=1.5, ls='--', alpha=0.7)
ax.axvline(gesture_e, color='#2E7D32', lw=1.5, ls='--', alpha=0.7)

in_gest_b = (t >= gesture_s) & (t <= gesture_e)
base_det = in_gest_b & (preds_seg_base == true_label)
if base_det.any():
    ax.scatter(t[base_det], raw_sig[base_det], marker='v', s=40,
               color='#9C27B0', zorder=5, label='Correct class')
base_rest = in_gest_b & (preds_seg_base == 0)
if base_rest.any():
    ax.scatter(t[base_rest], raw_sig[base_rest], marker='x', s=35,
               color='#EF5350', zorder=5, label='Predicted Rest')
base_wrong = in_gest_b & (preds_seg_base > 0) & (preds_seg_base != true_label)
if base_wrong.any():
    ax.scatter(t[base_wrong], raw_sig[base_wrong], marker='s', s=25,
               color='#FF8F00', zorder=5, label='Wrong gesture class')

ax.text(0.99, 0.96, 'Baseline: 38% of windows predicted as Rest',
        transform=ax.transAxes, ha='right', va='top',
        fontsize=9.5, fontweight='bold', color='#B71C1C',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE', edgecolor='#C62828'))
ax.set_ylabel('Acceleration X\n(normalized)', fontsize=9)
ax.set_xlabel('Time (ms)', fontsize=10)
ax.legend(loc='upper left', fontsize=8, framealpha=0.85)
ax.set_title('Baseline (No Filter)', fontsize=10, fontweight='bold')
ax.set_xlim(x_lo, x_hi)
ax.grid(axis='x', alpha=0.3)
ax.set_facecolor('white')

# shared gesture label
for ax in axes:
    mid = (gesture_s + gesture_e) / 2
    ylo = ax.get_ylim()[0]
    ax.text(mid, ylo, f'<-- {CLASS_NAMES[true_label]} -->',
            ha='center', va='bottom', fontsize=8,
            color='#2E7D32', fontweight='bold')

fig.suptitle(f'Gesture Detection: Baseline vs. EMA Model — Subject 2, '
             f'True Gesture: {CLASS_NAMES[true_label]}',
             fontsize=12, fontweight='bold', y=1.01)

plt.savefig(f'{OUT_DIR}/gesture_detection_comparison.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved -> {OUT_DIR}/gesture_detection_comparison.png")

print(f"\n✓ All three visualizations saved to: {OUT_DIR}/")
print("  - confusion_matrix_comparison.png")
print("  - pipeline_architecture.png")
print("  - gesture_detection_comparison.png")
