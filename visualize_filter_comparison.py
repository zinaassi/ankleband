"""
Visualize the comprehensive filter comparison results
Creates plots showing best config from each filter type
"""

import sys
sys.path.append('.')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from comprehensive_filter_comparison import *

def create_comparison_plots(h5_file, subject_name, output_dir='filter_comparison_results'):
    """Create visual comparison of all 6 filter types (best configs only)"""

    print(f'\n{"="*80}')
    print(f'Creating visual comparison plots for {subject_name}')
    print(f'{"="*80}\n')

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

    # Load best configurations from CSV
    results_csv = f'{output_dir}/{subject_name}/all_filters_comparison.csv'
    df_results = pd.read_csv(results_csv)

    # Get best config for each filter type
    best_configs = {}
    for filter_type in df_results['filter_type'].unique():
        type_results = df_results[df_results['filter_type'] == filter_type]
        best = type_results.loc[type_results['composite_score'].idxmax()]
        best_configs[filter_type] = best

    # ========================================================================
    # Apply best filters
    # ========================================================================

    filtered_signals = {}

    # Single-Pole IIR
    best = best_configs['Single-Pole IIR']
    alpha = eval(best['params'])['alpha']
    filtered_signals['Single-Pole IIR'] = {
        'signal': single_pole_iir(raw_signal, alpha),
        'name': f'IIR α={alpha:.2f}',
        'metrics': best
    }

    # Moving Average
    best = best_configs['Moving Average']
    window = eval(best['params'])['window']
    filtered_signals['Moving Average'] = {
        'signal': moving_average_filter(raw_signal, window),
        'name': f'MAF window={window}',
        'metrics': best
    }

    # Butterworth
    best = best_configs['Butterworth']
    params = eval(best['params'])
    filtered_signals['Butterworth'] = {
        'signal': butterworth_lowpass(raw_signal, params['cutoff_hz'], params['order']),
        'name': f'Butter {params["cutoff_hz"]}Hz O{params["order"]}',
        'metrics': best
    }

    # Biquad
    best = best_configs['Biquad']
    params = eval(best['params'])
    filtered_signals['Biquad'] = {
        'signal': biquad_lowpass(raw_signal, params['cutoff_hz'], params['Q']),
        'name': f'Biquad {params["cutoff_hz"]}Hz Q={params["Q"]:.2f}',
        'metrics': best
    }

    # Complementary
    best = best_configs['Complementary']
    alpha = eval(best['params'])['alpha']
    filtered_signals['Complementary'] = {
        'signal': complementary_filter(raw_signal, gyro_signal, alpha),
        'name': f'Comp α={alpha:.2f}',
        'metrics': best
    }

    # Kalman
    best = best_configs['Kalman']
    params = eval(best['params'])
    filtered_signals['Kalman'] = {
        'signal': kalman_filter(raw_signal, params['process_noise'], params['measurement_noise']),
        'name': f'Kalman Q={params["process_noise"]:.3f} R={params["measurement_noise"]:.1f}',
        'metrics': best
    }

    # ========================================================================
    # Create multi-subplot comparison figure
    # ========================================================================

    fig = plt.figure(figsize=(22, 16))
    fig.suptitle(f'Filter Type Comparison - Best Configuration of Each Type\\n{subject_name}',
                 fontsize=18, fontweight='bold', y=0.995)

    colors = {
        'Single-Pole IIR': 'blue',
        'Moving Average': 'green',
        'Butterworth': 'red',
        'Biquad': 'purple',
        'Complementary': 'orange',
        'Kalman': 'brown'
    }

    # Plot 1: Raw signal
    ax0 = plt.subplot(7, 1, 1)
    ax0.plot(time_axis, raw_signal, color='black', linewidth=2, label='Raw Signal (No Filter)')

    # Highlight gestures
    if gesture_mask.any():
        for j in range(len(gesture_mask)-1):
            if gesture_mask[j] and not gesture_mask[j-1]:
                gesture_start = time_axis[j]
            if gesture_mask[j] and not gesture_mask[j+1]:
                gesture_end = time_axis[j]
                ax0.axvspan(gesture_start, gesture_end, alpha=0.15, color='green')

    ax0.set_ylabel('acc_x (m/s²)', fontsize=11, fontweight='bold')
    ax0.set_title('Raw Signal (Baseline)', fontsize=12, fontweight='bold', loc='left')
    ax0.grid(True, alpha=0.3)
    ax0.set_xlim([time_axis[0], time_axis[-1]])
    ax0.legend(loc='upper right')

    # Plot 2-7: Each filter type
    for idx, (filter_type, data) in enumerate(filtered_signals.items(), 2):
        ax = plt.subplot(7, 1, idx)

        # Plot filtered signal
        ax.plot(time_axis, data['signal'], color=colors[filter_type], linewidth=2.5,
                label=f'{filter_type}: {data["name"]}')

        # Highlight gestures
        if gesture_mask.any():
            for j in range(len(gesture_mask)-1):
                if gesture_mask[j] and not gesture_mask[j-1]:
                    gesture_start = time_axis[j]
                if gesture_mask[j] and not gesture_mask[j+1]:
                    gesture_end = time_axis[j]
                    ax.axvspan(gesture_start, gesture_end, alpha=0.15, color='green')

        ax.set_ylabel('acc_x (m/s²)', fontsize=11, fontweight='bold')
        ax.set_title(f'{filter_type}: {data["name"]}', fontsize=12, fontweight='bold', loc='left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([time_axis[0], time_axis[-1]])
        ax.legend(loc='upper right')

        # Add metrics box
        metrics = data['metrics']
        peak_loss = metrics['peak_loss']

        if peak_loss < 10:
            verdict = 'EXCELLENT ✓✓'
            box_color = 'lightgreen'
        elif peak_loss < 20:
            verdict = 'GOOD ✓'
            box_color = 'lightyellow'
        else:
            verdict = 'ACCEPTABLE'
            box_color = 'lightcoral'

        metrics_text = (
            f'{verdict}\\n'
            f'Score: {metrics["composite_score"]:.3f}\\n'
            f'Peak: {metrics["peak_preserved"]:.1f}%\\n'
            f'Noise: {metrics["noise_reduction"]:.1f}%\\n'
            f'Delay: {metrics["delay_ms"]:.1f}ms'
        )

        ax.text(0.02, 0.97, metrics_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.85))

    plt.xlabel('Time (seconds)', fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    filename = f'{output_dir}/{subject_name}/filter_type_comparison_visual.png'
    plt.savefig(filename, dpi=250, bbox_inches='tight')
    plt.close()
    print(f'✅ Saved: {filename}')

    # ========================================================================
    # Create metrics comparison bar charts
    # ========================================================================

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'Filter Performance Metrics Comparison\\n{subject_name}',
                 fontsize=16, fontweight='bold')

    filter_names = list(filtered_signals.keys())
    filter_colors = [colors[name] for name in filter_names]

    # Extract metrics
    peak_preserved = [filtered_signals[name]['metrics']['peak_preserved'] for name in filter_names]
    noise_reduction = [filtered_signals[name]['metrics']['noise_reduction'] for name in filter_names]
    correlation = [filtered_signals[name]['metrics']['correlation'] * 100 for name in filter_names]
    delay_ms = [filtered_signals[name]['metrics']['delay_ms'] for name in filter_names]
    composite_score = [filtered_signals[name]['metrics']['composite_score'] * 100 for name in filter_names]
    peak_loss = [filtered_signals[name]['metrics']['peak_loss'] for name in filter_names]

    # Plot 1: Peak Preserved (higher is better)
    ax = axes[0, 0]
    bars = ax.bar(range(len(filter_names)), peak_preserved, color=filter_colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=100, color='green', linestyle='--', linewidth=2, label='100% (Perfect)')
    ax.set_ylabel('Peak Preserved (%)', fontsize=12, fontweight='bold')
    ax.set_title('Peak Preservation (Higher is Better)', fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(filter_names)))
    ax.set_xticklabels(filter_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{peak_preserved[i]:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

    # Plot 2: Noise Reduction (higher is better)
    ax = axes[0, 1]
    bars = ax.bar(range(len(filter_names)), noise_reduction, color=filter_colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Noise Reduction (%)', fontsize=12, fontweight='bold')
    ax.set_title('Noise Reduction (Higher is Better)', fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(filter_names)))
    ax.set_xticklabels(filter_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{noise_reduction[i]:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

    # Plot 3: Correlation (higher is better)
    ax = axes[0, 2]
    bars = ax.bar(range(len(filter_names)), correlation, color=filter_colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=100, color='green', linestyle='--', linewidth=2, label='100% (Perfect)')
    ax.set_ylabel('Correlation (%)', fontsize=12, fontweight='bold')
    ax.set_title('Waveform Similarity (Higher is Better)', fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(filter_names)))
    ax.set_xticklabels(filter_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{correlation[i]:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

    # Plot 4: Delay (lower is better)
    ax = axes[1, 0]
    bars = ax.bar(range(len(filter_names)), delay_ms, color=filter_colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Delay (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Processing Delay (Lower is Better)', fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(filter_names)))
    ax.set_xticklabels(filter_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{delay_ms[i]:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

    # Plot 5: Peak Loss (lower is better)
    ax = axes[1, 1]
    bars = ax.bar(range(len(filter_names)), peak_loss, color=filter_colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=10, color='green', linestyle='--', linewidth=2, label='<10% (Excellent)')
    ax.axhline(y=20, color='orange', linestyle='--', linewidth=2, label='<20% (Good)')
    ax.set_ylabel('Peak Loss (%)', fontsize=12, fontweight='bold')
    ax.set_title('Peak Loss (Lower is Better)', fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(filter_names)))
    ax.set_xticklabels(filter_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=9)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{peak_loss[i]:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

    # Plot 6: Composite Score (higher is better)
    ax = axes[1, 2]
    bars = ax.bar(range(len(filter_names)), composite_score, color=filter_colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Composite Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Overall Performance Score (Higher is Better)', fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(filter_names)))
    ax.set_xticklabels(filter_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{composite_score[i]:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    filename = f'{output_dir}/{subject_name}/filter_metrics_comparison.png'
    plt.savefig(filename, dpi=250, bbox_inches='tight')
    plt.close()
    print(f'✅ Saved: {filename}')

    print(f'\\n{"="*80}')
    print('✅ All comparison plots created successfully!')
    print(f'{"="*80}\\n')

if __name__ == '__main__':
    h5_file = 'data/dataset/ID01_seating_all_gestures.h5'
    subject_name = 'ID01_Seating'

    create_comparison_plots(h5_file, subject_name, output_dir='filter_comparison_results')
