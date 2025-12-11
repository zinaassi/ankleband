"""
Filter Comparison Based on Research Paper Requirements

Key metrics for the paper:
1. Signal distribution preservation (for batch normalization)
2. Temporal feature preservation (for 1D Conv layer)
3. Real-time responsiveness (signal delay)
4. Computational efficiency (ESP32 power consumption)
5. Peak preservation (gesture detection)
6. Noise characteristics (CNN trained on noisy data)
"""

import sys
sys.path.append('.')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from comprehensive_filter_comparison import *
import os

def calculate_research_metrics(raw, filtered, gesture_mask, sampling_rate=200):
    """
    Calculate metrics relevant to the research paper:
    - Distribution similarity (critical for batch norm)
    - Temporal feature preservation (for 1D Conv)
    - Signal delay (user experience)
    - Peak preservation (gesture detection)
    """

    # 1. Peak Preservation (gesture detection accuracy)
    if gesture_mask.any():
        raw_peak = np.max(np.abs(raw[gesture_mask]))
        filt_peak = np.max(np.abs(filtered[gesture_mask]))
    else:
        raw_peak = np.max(np.abs(raw))
        filt_peak = np.max(np.abs(filtered))

    peak_preserved_pct = (filt_peak / raw_peak) * 100
    peak_loss_pct = 100 - peak_preserved_pct

    # 2. Distribution Similarity (critical for batch normalization)
    # Measure using statistical moments
    raw_mean = np.mean(raw)
    filt_mean = np.mean(filtered)
    raw_std = np.std(raw)
    filt_std = np.std(filtered)
    raw_skew = np.mean((raw - raw_mean)**3) / (raw_std**3) if raw_std > 0 else 0
    filt_skew = np.mean((filtered - filt_mean)**3) / (filt_std**3) if filt_std > 0 else 0

    mean_diff = abs(raw_mean - filt_mean)
    std_diff = abs(raw_std - filt_std)
    skew_diff = abs(raw_skew - filt_skew)

    # Distribution similarity score (0 to 1, higher is better)
    dist_similarity = 1.0 / (1.0 + mean_diff + std_diff + skew_diff)

    # 3. Waveform Correlation (temporal feature preservation)
    correlation = np.corrcoef(raw, filtered)[0, 1]

    # 4. Signal Delay (using cross-correlation)
    cross_corr = np.correlate(raw, filtered, mode='full')
    delay_samples = len(raw) - 1 - np.argmax(cross_corr)
    delay_ms = (delay_samples / sampling_rate) * 1000

    # 5. Noise Reduction (should be gentle, not aggressive)
    non_gesture_mask = ~gesture_mask
    if non_gesture_mask.any():
        raw_noise_std = np.std(raw[non_gesture_mask])
        filt_noise_std = np.std(filtered[non_gesture_mask])
        noise_reduction_pct = (1 - filt_noise_std / raw_noise_std) * 100
    else:
        noise_reduction_pct = 0

    # 6. Signal Energy Preservation (for CNN feature extraction)
    raw_energy = np.sum(raw**2)
    filt_energy = np.sum(filtered**2)
    energy_preserved_pct = (filt_energy / raw_energy) * 100

    # 7. High-frequency content preservation (temporal features for Conv1D)
    # Use difference between consecutive samples as proxy for high-freq content
    raw_diff = np.diff(raw)
    filt_diff = np.diff(filtered)
    raw_hf_energy = np.sum(raw_diff**2)
    filt_hf_energy = np.sum(filt_diff**2)
    hf_preserved_pct = (filt_hf_energy / raw_hf_energy) * 100 if raw_hf_energy > 0 else 0

    # 8. Research Paper Score (weighted for paper priorities)
    # Priorities based on paper:
    #   - Distribution similarity: 30% (batch norm generalization)
    #   - Peak preservation: 25% (gesture detection)
    #   - Temporal features: 20% (1D Conv)
    #   - Low delay: 15% (user experience)
    #   - Gentle noise reduction: 10% (preserve CNN training characteristics)

    # Normalize delay (0-10ms = 1.0, >20ms = 0.0)
    delay_score = max(0, 1 - abs(delay_ms) / 20.0)

    # Normalize noise reduction (5-15% = best, 0% or >30% = worse)
    if 5 <= noise_reduction_pct <= 15:
        noise_score = 1.0
    elif noise_reduction_pct < 5:
        noise_score = noise_reduction_pct / 5.0
    else:
        noise_score = max(0, 1 - (noise_reduction_pct - 15) / 30.0)

    # Peak preservation (95-105% = best)
    if 95 <= peak_preserved_pct <= 105:
        peak_score = 1.0
    elif peak_preserved_pct < 95:
        peak_score = peak_preserved_pct / 95.0
    else:  # >105% means amplification
        peak_score = max(0, 1 - (peak_preserved_pct - 105) / 50.0)

    research_score = (
        dist_similarity * 0.30 +
        peak_score * 0.25 +
        correlation * 0.20 +
        delay_score * 0.15 +
        noise_score * 0.10
    )

    return {
        'peak_preserved_pct': peak_preserved_pct,
        'peak_loss_pct': peak_loss_pct,
        'distribution_similarity': dist_similarity,
        'waveform_correlation': correlation,
        'signal_delay_ms': abs(delay_ms),
        'noise_reduction_pct': noise_reduction_pct,
        'energy_preserved_pct': energy_preserved_pct,
        'hf_content_preserved_pct': hf_preserved_pct,
        'research_score': research_score,
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        'skew_diff': skew_diff
    }

def create_research_paper_visualization(h5_file, subject_name, output_dir='filter_comparison_results'):
    """
    Create visualization matching research paper requirements:
    - Each filter in separate subplot
    - All filters on same graph with raw data
    - Highlight gesture regions
    - Show research-relevant metrics
    """

    print(f'\n{"="*80}')
    print(f'Creating Research Paper Visualizations for {subject_name}')
    print(f'{"="*80}\n')

    # Create output directory
    os.makedirs(f'{output_dir}/{subject_name}', exist_ok=True)

    # Load data
    df = pd.read_hdf(h5_file, key='df')

    sampling_rate = 200
    start_time = 30
    duration = 5
    start_idx = int(start_time * sampling_rate)
    end_idx = int((start_time + duration) * sampling_rate)
    df_window = df.iloc[start_idx:end_idx].copy()

    time_axis = np.arange(len(df_window)) / sampling_rate
    raw_signal = df_window['acc_x'].values
    gesture_mask = df_window['label'].values > 0
    gyro_signal = df_window['gyro_x'].values

    # Apply best filters from each type
    filters = {
        'Moving Average (5)': moving_average_filter(raw_signal, 5),
        'Single-Pole IIR (α=0.40)': single_pole_iir(raw_signal, 0.40),
        'Butterworth (40Hz O1)': butterworth_lowpass(raw_signal, 40, 1),
        'Biquad (30Hz Q=1.50)': biquad_lowpass(raw_signal, 30, 1.5),
        'Kalman (Q=0.05 R=0.1)': kalman_filter(raw_signal, 0.05, 0.1),
        'Complementary (α=0.80)': complementary_filter(raw_signal, gyro_signal, 0.8)
    }

    # Calculate research metrics for each filter
    research_metrics = {}
    for name, filtered in filters.items():
        research_metrics[name] = calculate_research_metrics(raw_signal, filtered, gesture_mask)

    # ========================================================================
    # FIGURE 1: Individual Subplots (Raw + Each Filter)
    # ========================================================================

    fig = plt.figure(figsize=(22, 18))
    fig.suptitle(f'Filter Comparison for Research Paper - {subject_name}\\n' +
                 f'Raw Signal + 6 Filter Types (Best Configuration Each)',
                 fontsize=18, fontweight='bold', y=0.995)

    colors = {
        'Moving Average (5)': '#2ecc71',  # Green
        'Single-Pole IIR (α=0.40)': '#3498db',  # Blue
        'Butterworth (40Hz O1)': '#e74c3c',  # Red
        'Biquad (30Hz Q=1.50)': '#9b59b6',  # Purple
        'Kalman (Q=0.05 R=0.1)': '#f39c12',  # Orange
        'Complementary (α=0.80)': '#34495e'  # Dark gray
    }

    # Subplot 1: Raw Signal
    ax0 = plt.subplot(7, 1, 1)
    ax0.plot(time_axis, raw_signal, color='black', linewidth=2.5, label='Raw Signal (Baseline)', alpha=0.8)

    if gesture_mask.any():
        for j in range(len(gesture_mask)-1):
            if gesture_mask[j] and not gesture_mask[j-1]:
                gesture_start = time_axis[j]
            if gesture_mask[j] and not gesture_mask[j+1]:
                gesture_end = time_axis[j]
                ax0.axvspan(gesture_start, gesture_end, alpha=0.15, color='green', label='Gesture' if j == 1 else '')

    ax0.set_ylabel('Acceleration\\n(m/s²)', fontsize=12, fontweight='bold')
    ax0.set_title('RAW SIGNAL (No Filter) - CNN Training Data Distribution', fontsize=13, fontweight='bold', loc='left')
    ax0.grid(True, alpha=0.3)
    ax0.set_xlim([time_axis[0], time_axis[-1]])
    ax0.legend(loc='upper right', fontsize=10)

    # Add raw signal statistics
    raw_text = f'Mean: {np.mean(raw_signal):.3f}\\nStd: {np.std(raw_signal):.3f}'
    ax0.text(0.98, 0.95, raw_text, transform=ax0.transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    # Subplots 2-7: Each Filter
    for idx, (name, filtered) in enumerate(filters.items(), 2):
        ax = plt.subplot(7, 1, idx)

        # Plot filtered signal
        ax.plot(time_axis, filtered, color=colors[name], linewidth=2.5, label=name, alpha=0.9)

        # Overlay raw signal (faint)
        ax.plot(time_axis, raw_signal, color='black', linewidth=1, alpha=0.15, linestyle='--', label='Raw (reference)')

        # Highlight gesture regions
        if gesture_mask.any():
            for j in range(len(gesture_mask)-1):
                if gesture_mask[j] and not gesture_mask[j-1]:
                    gesture_start = time_axis[j]
                if gesture_mask[j] and not gesture_mask[j+1]:
                    gesture_end = time_axis[j]
                    ax.axvspan(gesture_start, gesture_end, alpha=0.15, color='green')

        ax.set_ylabel('Acceleration\\n(m/s²)', fontsize=12, fontweight='bold')
        ax.set_title(name, fontsize=13, fontweight='bold', loc='left', color=colors[name])
        ax.grid(True, alpha=0.3)
        ax.set_xlim([time_axis[0], time_axis[-1]])
        ax.legend(loc='upper right', fontsize=10)

        # Add research metrics
        metrics = research_metrics[name]

        # Color code based on research score
        if metrics['research_score'] >= 0.80:
            verdict = 'EXCELLENT'
            box_color = 'lightgreen'
        elif metrics['research_score'] >= 0.70:
            verdict = 'GOOD'
            box_color = 'lightyellow'
        elif metrics['research_score'] >= 0.60:
            verdict = 'ACCEPTABLE'
            box_color = 'lightcoral'
        else:
            verdict = 'POOR'
            box_color = 'salmon'

        metrics_text = (
            f'{verdict}\\n'
            f'Research Score: {metrics["research_score"]:.3f}\\n'
            f'Peak: {metrics["peak_preserved_pct"]:.1f}%\\n'
            f'Dist. Sim: {metrics["distribution_similarity"]:.3f}\\n'
            f'Correlation: {metrics["waveform_correlation"]:.3f}\\n'
            f'Delay: {metrics["signal_delay_ms"]:.1f}ms\\n'
            f'Noise: {metrics["noise_reduction_pct"]:.1f}%'
        )

        ax.text(0.02, 0.97, metrics_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.85))

    ax.set_xlabel('Time (seconds)', fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    filename1 = f'{output_dir}/{subject_name}/research_paper_subplots.png'
    plt.savefig(filename1, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'✅ Saved: {filename1}')

    # ========================================================================
    # FIGURE 2: All Filters on Same Graph
    # ========================================================================

    fig, ax = plt.subplots(figsize=(20, 10))
    fig.suptitle(f'All Filters Comparison on Same Graph - {subject_name}\\n' +
                 f'Raw Signal vs. 6 Filter Types',
                 fontsize=16, fontweight='bold')

    # Plot raw signal (thick black line)
    ax.plot(time_axis, raw_signal, color='black', linewidth=3, label='Raw Signal (CNN Training)', alpha=0.7, zorder=10)

    # Plot all filtered signals
    linestyles = ['-', '--', '-.', ':', '-', '--']
    for (name, filtered), ls in zip(filters.items(), linestyles):
        ax.plot(time_axis, filtered, color=colors[name], linewidth=2,
                linestyle=ls, label=name, alpha=0.8)

    # Highlight gesture regions
    if gesture_mask.any():
        for j in range(len(gesture_mask)-1):
            if gesture_mask[j] and not gesture_mask[j-1]:
                gesture_start = time_axis[j]
            if gesture_mask[j] and not gesture_mask[j+1]:
                gesture_end = time_axis[j]
                ax.axvspan(gesture_start, gesture_end, alpha=0.15, color='green', zorder=1)
        # Add legend entry for gesture region
        ax.axvspan(0, 0, alpha=0.15, color='green', label='Gesture Region')

    ax.set_xlabel('Time (seconds)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Acceleration (m/s²)', fontsize=14, fontweight='bold')
    ax.set_title('Direct Comparison: How Each Filter Affects Raw Signal', fontsize=14, fontweight='bold', loc='left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([time_axis[0], time_axis[-1]])
    ax.legend(loc='upper right', fontsize=11, ncol=2)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    filename2 = f'{output_dir}/{subject_name}/research_paper_overlay.png'
    plt.savefig(filename2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'✅ Saved: {filename2}')

    # ========================================================================
    # FIGURE 3: Research Metrics Comparison
    # ========================================================================

    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    fig.suptitle(f'Research-Relevant Metrics Comparison - {subject_name}',
                 fontsize=16, fontweight='bold')

    filter_names = list(filters.keys())
    filter_colors_list = [colors[name] for name in filter_names]

    # Extract metrics
    research_scores = [research_metrics[name]['research_score'] * 100 for name in filter_names]
    peak_preserved = [research_metrics[name]['peak_preserved_pct'] for name in filter_names]
    dist_similarity = [research_metrics[name]['distribution_similarity'] * 100 for name in filter_names]
    correlation = [research_metrics[name]['waveform_correlation'] * 100 for name in filter_names]
    delay_ms = [research_metrics[name]['signal_delay_ms'] for name in filter_names]
    noise_reduction = [research_metrics[name]['noise_reduction_pct'] for name in filter_names]

    # Plot 1: Research Score (MOST IMPORTANT)
    ax = axes[0, 0]
    bars = ax.bar(range(len(filter_names)), research_scores, color=filter_colors_list, alpha=0.7, edgecolor='black', linewidth=2)
    ax.axhline(y=80, color='green', linestyle='--', linewidth=2, label='Excellent (≥80)')
    ax.axhline(y=70, color='orange', linestyle='--', linewidth=2, label='Good (≥70)')
    ax.set_ylabel('Research Score (%)', fontsize=13, fontweight='bold')
    ax.set_title('OVERALL: Research Paper Score\\n(Dist.Sim 30% + Peak 25% + Corr 20% + Delay 15% + Noise 10%)',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(filter_names)))
    ax.set_xticklabels([name.split('(')[0].strip() for name in filter_names], rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=9)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{research_scores[i]:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Plot 2: Distribution Similarity (CRITICAL for Batch Norm)
    ax = axes[0, 1]
    bars = ax.bar(range(len(filter_names)), dist_similarity, color=filter_colors_list, alpha=0.7, edgecolor='black', linewidth=2)
    ax.axhline(y=90, color='green', linestyle='--', linewidth=2, label='Excellent (≥90%)')
    ax.set_ylabel('Distribution Similarity (%)', fontsize=13, fontweight='bold')
    ax.set_title('Distribution Similarity\\n(Critical for Batch Normalization Generalization)',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(filter_names)))
    ax.set_xticklabels([name.split('(')[0].strip() for name in filter_names], rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=9)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{dist_similarity[i]:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Plot 3: Waveform Correlation (temporal features)
    ax = axes[0, 2]
    bars = ax.bar(range(len(filter_names)), correlation, color=filter_colors_list, alpha=0.7, edgecolor='black', linewidth=2)
    ax.axhline(y=95, color='green', linestyle='--', linewidth=2, label='Excellent (≥95%)')
    ax.set_ylabel('Waveform Correlation (%)', fontsize=13, fontweight='bold')
    ax.set_title('Temporal Feature Preservation\\n(For 1D Convolutional Layer)',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(filter_names)))
    ax.set_xticklabels([name.split('(')[0].strip() for name in filter_names], rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=9)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{correlation[i]:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Plot 4: Peak Preservation (gesture detection)
    ax = axes[1, 0]
    bars = ax.bar(range(len(filter_names)), peak_preserved, color=filter_colors_list, alpha=0.7, edgecolor='black', linewidth=2)
    ax.axhline(y=100, color='green', linestyle='--', linewidth=2, label='Perfect (100%)')
    ax.axhline(y=95, color='orange', linestyle='--', linewidth=1.5, label='Good (≥95%)')
    ax.set_ylabel('Peak Preserved (%)', fontsize=13, fontweight='bold')
    ax.set_title('Peak Preservation\\n(Gesture Detection Accuracy)',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(filter_names)))
    ax.set_xticklabels([name.split('(')[0].strip() for name in filter_names], rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=9)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{peak_preserved[i]:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Plot 5: Signal Delay (user experience)
    ax = axes[1, 1]
    bars = ax.bar(range(len(filter_names)), delay_ms, color=filter_colors_list, alpha=0.7, edgecolor='black', linewidth=2)
    ax.axhline(y=10, color='orange', linestyle='--', linewidth=2, label='Acceptable (<10ms)')
    ax.set_ylabel('Signal Delay (ms)', fontsize=13, fontweight='bold')
    ax.set_title('Signal Group Delay\\n(Real-time Responsiveness)',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(filter_names)))
    ax.set_xticklabels([name.split('(')[0].strip() for name in filter_names], rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=9)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{delay_ms[i]:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Plot 6: Noise Reduction (should be gentle)
    ax = axes[1, 2]
    bars = ax.bar(range(len(filter_names)), noise_reduction, color=filter_colors_list, alpha=0.7, edgecolor='black', linewidth=2)
    ax.axhline(y=15, color='green', linestyle='--', linewidth=2, label='Ideal (5-15%)')
    ax.axhline(y=5, color='green', linestyle='--', linewidth=2)
    ax.set_ylabel('Noise Reduction (%)', fontsize=13, fontweight='bold')
    ax.set_title('Noise Reduction\\n(Gentle Filtering Preferred - CNN Handles Noise)',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(filter_names)))
    ax.set_xticklabels([name.split('(')[0].strip() for name in filter_names], rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=9)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{noise_reduction[i]:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    filename3 = f'{output_dir}/{subject_name}/research_metrics_comparison.png'
    plt.savefig(filename3, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'✅ Saved: {filename3}')

    # ========================================================================
    # Save Research Metrics to CSV
    # ========================================================================

    metrics_data = []
    for name in filter_names:
        m = research_metrics[name]
        metrics_data.append({
            'Filter': name,
            'Research_Score': m['research_score'],
            'Peak_Preserved_%': m['peak_preserved_pct'],
            'Distribution_Similarity': m['distribution_similarity'],
            'Waveform_Correlation': m['waveform_correlation'],
            'Signal_Delay_ms': m['signal_delay_ms'],
            'Noise_Reduction_%': m['noise_reduction_pct'],
            'Energy_Preserved_%': m['energy_preserved_pct'],
            'HF_Content_Preserved_%': m['hf_content_preserved_pct']
        })

    df_metrics = pd.DataFrame(metrics_data)
    df_metrics = df_metrics.sort_values('Research_Score', ascending=False)

    csv_file = f'{output_dir}/{subject_name}/research_metrics.csv'
    df_metrics.to_csv(csv_file, index=False)
    print(f'✅ Saved: {csv_file}')

    # ========================================================================
    # Print Summary
    # ========================================================================

    print(f'\n{"="*80}')
    print(f'RESEARCH METRICS SUMMARY')
    print(f'{"="*80}\n')

    print(f'{"Filter":<30} {"Score":<10} {"Dist.Sim":<10} {"Peak%":<10} {"Delay(ms)":<10}')
    print('-'*80)

    for _, row in df_metrics.iterrows():
        print(f'{row["Filter"]:<30} {row["Research_Score"]:.3f}      '
              f'{row["Distribution_Similarity"]:.3f}      '
              f'{row["Peak_Preserved_%"]:.1f}%     '
              f'{row["Signal_Delay_ms"]:.1f}')

    print(f'\n{"="*80}')
    print(f'🏆 BEST FOR RESEARCH PAPER: {df_metrics.iloc[0]["Filter"]}')
    print(f'   Research Score: {df_metrics.iloc[0]["Research_Score"]:.3f}')
    print(f'{"="*80}\n')

    return df_metrics

if __name__ == '__main__':
    h5_file = 'data/dataset/ID01_seating_all_gestures.h5'
    subject_name = 'ID01_Seating'

    df_metrics = create_research_paper_visualization(
        h5_file, subject_name,
        output_dir='filter_comparison_results'
    )

    print('\n✅ All research paper visualizations created!')
