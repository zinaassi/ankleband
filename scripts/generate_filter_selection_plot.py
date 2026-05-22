#!/usr/bin/env python3
"""
Filter selection justification plot.
Two subplots:
  1. Horizontal bar chart of all 82 configs ranked by revised deployment score
  2. Grouped bar chart of key metrics for the 7 selected configs
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

OUT_DIR = "outputs/report_visualizations"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load raw metrics ────────────────────────────────────────────────────────
df = pd.read_csv('outputs/filter_comparison_esp32/results_tables/all_filters_all_metrics.csv')

# Average metrics per config across signals
grouped = df.groupby(['filter_type', 'config_label']).agg(
    snr_db=('snr_db', 'mean'),
    peak_pres=('peak_preservation', 'mean'),
    edge_sharp=('edge_sharpness', 'mean'),
    phase_delay_ms=('phase_delay_ms', 'mean'),
    correlation=('correlation', 'mean'),
).reset_index()

# ── Revised deployment score (same weights as DEPLOYMENT_REPORT.txt) ────────
def minmax(s):
    return (s - s.min()) / (s.max() - s.min() + 1e-9)

grouped['nr_norm'] = minmax(grouped['snr_db'])                     # noise reduction
grouped['pp_norm'] = minmax(grouped['peak_pres'].clip(0, 1))       # peak preservation
grouped['es_norm'] = minmax(grouped['edge_sharp'])                  # edge sharpness
grouped['pd_norm'] = 1 - minmax(grouped['phase_delay_ms'])         # phase delay (lower=better)
grouped['sc_norm'] = minmax(grouped['correlation'].clip(0, 1))     # shape correlation

grouped['deployment_score'] = 100 * (
    0.30 * grouped['nr_norm'] +
    0.28 * grouped['pp_norm'] +
    0.22 * grouped['es_norm'] +
    0.12 * grouped['pd_norm'] +
    0.08 * grouped['sc_norm']
)

ranked = grouped.sort_values('deployment_score', ascending=False).reset_index(drop=True)
ranked['rank'] = ranked.index + 1

# ── 7 configs selected for CNN training ─────────────────────────────────────
# (filter_type, config_label, display_name)
SELECTED = [
    ('EMA',         '\u03b1=0.3',        'EMA \u03b1=0.3'),
    ('EMA',         '\u03b1=0.5',        'EMA \u03b1=0.5'),
    ('Butterworth', '40Hz-O2',           'Butterworth 40Hz-O2'),
    ('Biquad',      '30Hz-Q1.0',         'Biquad 30Hz-Q1.0'),
    ('Kalman',      'Q=0.0001-R=0.0001', 'Kalman Q=0.0001/R=0.0001'),
    ('Kalman',      'Q=0.1-R=0.1',       'Kalman Q=0.1/R=0.1'),
    ('Kalman',      'Q=0.001-R=0.001',   'Kalman Q=0.001/R=0.001'),
]

sel_keys = set((ft, cfg) for ft, cfg, _ in SELECTED)
ranked['selected'] = ranked.apply(
    lambda r: (r['filter_type'], r['config_label']) in sel_keys, axis=1)

# Print rank info for written context
print("=== SELECTED CONFIG RANKS ===")
for ft, cfg, disp in SELECTED:
    row = ranked[(ranked['filter_type'] == ft) & (ranked['config_label'] == cfg)]
    if not row.empty:
        r = row.iloc[0]
        print(f"  [{r['rank']:2d}/82] {disp:30s}  score={r['deployment_score']:.1f}  "
              f"NR={r['nr_norm']*100:.1f}%  PP={r['pp_norm']*100:.1f}%  "
              f"ES={r['es_norm']*100:.1f}%  PD={r['pd_norm']*100:.1f}%")
    else:
        print(f"  NOT FOUND: {ft} {cfg}")

# ── Type color map ───────────────────────────────────────────────────────────
TYPE_COLORS = {
    'EMA':           '#1565C0',
    'Butterworth':   '#2E7D32',
    'Biquad':        '#6A1B9A',
    'Kalman':        '#E65100',
    'MAF':           '#00695C',
    'Complementary': '#37474F',
}

# ── Figure ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 16), facecolor='white')
gs  = fig.add_gridspec(2, 1, height_ratios=[3, 1.2], hspace=0.42)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

# ── TOP SUBPLOT: horizontal bar chart, all 82 configs ───────────────────────
n = len(ranked)
y_pos = np.arange(n)

bar_colors = []
for _, row in ranked.iterrows():
    if row['selected']:
        bar_colors.append(TYPE_COLORS.get(row['filter_type'], '#333333'))
    else:
        bar_colors.append('#D0D0D0')

bars = ax1.barh(y_pos, ranked['deployment_score'], color=bar_colors,
                edgecolor='none', height=0.82)

# Cutoff line at rank of lowest-scoring selected config
sel_rows = ranked[ranked['selected']]
cutoff_score = sel_rows['deployment_score'].min()
ax1.axvline(cutoff_score, color='#B71C1C', lw=1.6, ls='--', zorder=5,
            label=f'Selection cutoff ({cutoff_score:.1f})')

# Label only selected configs
for _, row in sel_rows.iterrows():
    idx = row['rank'] - 1
    ax1.text(row['deployment_score'] + 0.5, idx,
             f"  {row['config_label']}",
             va='center', fontsize=7.5, color=TYPE_COLORS[row['filter_type']],
             fontweight='bold')

# y-axis: rank numbers, reversed so rank 1 is at top
ax1.set_yticks(y_pos[::5])
ax1.set_yticklabels([f"#{r}" for r in ranked['rank'].values[::5]], fontsize=7)
ax1.invert_yaxis()
ax1.set_xlabel('Revised Deployment Score\n(NR 30% + PP 28% + ES 22% + PD 12% + SC 8%)',
               fontsize=10)
ax1.set_title('All 82 Filter Configurations — Revised Deployment Score Ranking',
              fontsize=11, fontweight='bold')
ax1.set_xlim(0, 115)
ax1.grid(axis='x', alpha=0.3)
ax1.set_facecolor('white')

# Legend: filter families
legend_patches = [
    mpatches.Patch(color=TYPE_COLORS[ft], label=ft)
    for ft in TYPE_COLORS if ft in ranked['filter_type'].values
]
legend_patches += [
    mpatches.Patch(color='#D0D0D0', label='Not selected (78 configs)'),
    plt.Line2D([0], [0], color='#B71C1C', lw=1.6, ls='--', label='Selection cutoff'),
]
ax1.legend(handles=legend_patches, fontsize=8, loc='lower right',
           framealpha=0.9, ncol=2)

# ── BOTTOM SUBPLOT: key metrics for 7 selected configs + baseline ────────────
disp_names = [d for _, _, d in SELECTED]
sel_data = []
for ft, cfg, disp in SELECTED:
    row = ranked[(ranked['filter_type'] == ft) & (ranked['config_label'] == cfg)]
    if not row.empty:
        r = row.iloc[0]
        sel_data.append({
            'name':  disp,
            'ft':    ft,
            'NR':    r['nr_norm'] * 100,
            'PP':    r['pp_norm'] * 100,
            'ES':    r['es_norm'] * 100,
        })

sel_df = pd.DataFrame(sel_data)

x      = np.arange(len(sel_df))
width  = 0.25
METRIC_COLORS = ['#1565C0', '#2E7D32', '#F57F17']

bars_nr = ax2.bar(x - width, sel_df['NR'], width, color=METRIC_COLORS[0],
                  label='Noise Reduction', edgecolor='white', linewidth=0.8)
bars_pp = ax2.bar(x,          sel_df['PP'], width, color=METRIC_COLORS[1],
                  label='Peak Preservation', edgecolor='white', linewidth=0.8)
bars_es = ax2.bar(x + width, sel_df['ES'], width, color=METRIC_COLORS[2],
                  label='Edge Sharpness', edgecolor='white', linewidth=0.8)

# Value labels
for bars_set in [bars_nr, bars_pp, bars_es]:
    for bar in bars_set:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 1,
                 f'{h:.0f}', ha='center', va='bottom', fontsize=7)

ax2.set_xticks(x)
ax2.set_xticklabels(sel_df['name'], rotation=18, ha='right', fontsize=8.5)
ax2.set_ylabel('Normalized Score (%)', fontsize=10)
ax2.set_ylim(0, 115)
ax2.set_title('Selected 7 Configurations — Key Metric Comparison\n'
              '(shows diverse trade-off profiles justifying each selection)',
              fontsize=11, fontweight='bold')
ax2.legend(fontsize=9, framealpha=0.9)
ax2.grid(axis='y', alpha=0.3)
ax2.set_facecolor('white')

plt.savefig(f'{OUT_DIR}/filter_selection_justification.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\nSaved -> {OUT_DIR}/filter_selection_justification.png")

# ── Written context: per-family analysis ────────────────────────────────────
print("\n" + "="*70)
print("PER-FAMILY ANALYSIS FOR REPORT")
print("="*70)

for ft in ['EMA', 'Kalman', 'Butterworth', 'Biquad', 'MAF', 'Complementary']:
    fam = ranked[ranked['filter_type'] == ft].copy()
    if fam.empty:
        continue
    print(f"\n--- {ft} ({len(fam)} configs) ---")
    for _, r in fam.iterrows():
        sel_marker = " <-- SELECTED" if r['selected'] else ""
        print(f"  [{r['rank']:2d}] {r['config_label']:25s}  "
              f"score={r['deployment_score']:.1f}  "
              f"NR={r['nr_norm']*100:.0f}%  PP={r['pp_norm']*100:.0f}%  "
              f"ES={r['es_norm']*100:.0f}%  PD={r['pd_norm']*100:.0f}%"
              f"{sel_marker}")
