"""
Generate comparison visualization for 12Hz filter vs no filter
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Directories
no_filter_dir = 'outputs/cutoff_optimization/no_filter_baseline'
with_filter_dir = 'outputs/cutoff_optimization/cutoff_12Hz'
output_dir = 'outputs/comparison_12hz'

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Load metrics
no_filter_metrics = pd.read_csv(os.path.join(no_filter_dir, 'metrics.csv'))
with_filter_metrics = pd.read_csv(os.path.join(with_filter_dir, 'metrics.csv'))

# Load losses
no_filter_losses = pd.read_csv(os.path.join(no_filter_dir, 'losses.csv'))
with_filter_losses = pd.read_csv(os.path.join(with_filter_dir, 'losses.csv'))

# Print final metrics comparison
print('\n' + '='*80)
print('FINAL METRICS COMPARISON (Last Epoch)')
print('='*80)

metrics_comparison = pd.DataFrame({
    'Metric': ['Accuracy', 'Recall', 'Precision'],
    'No Filter': [
        no_filter_metrics['Accuracy'].iloc[-1],
        no_filter_metrics['Recall'].iloc[-1],
        no_filter_metrics['Precision'].iloc[-1]
    ],
    'With Filter (12Hz)': [
        with_filter_metrics['Accuracy'].iloc[-1],
        with_filter_metrics['Recall'].iloc[-1],
        with_filter_metrics['Precision'].iloc[-1]
    ]
})

# Calculate improvement
metrics_comparison['Improvement'] = (
    metrics_comparison['With Filter (12Hz)'] - metrics_comparison['No Filter']
) * 100

metrics_comparison['Improvement (%)'] = metrics_comparison['Improvement'].apply(
    lambda x: f'{x:+.2f}%'
)

print(metrics_comparison.to_string(index=False))
print('='*80)

# Save comparison table
comparison_file = os.path.join(output_dir, 'metrics_comparison.csv')
metrics_comparison.to_csv(comparison_file, index=False)
print(f'\n✅ Comparison table saved to: {comparison_file}')

# Create comprehensive comparison plots
fig = plt.figure(figsize=(18, 12))

# 1. Metrics over epochs
ax1 = plt.subplot(2, 3, 1)
epochs = range(1, len(no_filter_metrics) + 1)
ax1.plot(epochs, no_filter_metrics['Accuracy'], 'o-', color='red',
         linewidth=2, label='No Filter', markersize=6)
ax1.plot(epochs, with_filter_metrics['Accuracy'], 's-', color='blue',
         linewidth=2, label='With Filter (12Hz)', markersize=6)
ax1.set_xlabel('Epoch', fontweight='bold', fontsize=11)
ax1.set_ylabel('Accuracy', fontweight='bold', fontsize=11)
ax1.set_title('Accuracy over Epochs', fontweight='bold', fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

ax2 = plt.subplot(2, 3, 2)
ax2.plot(epochs, no_filter_metrics['Recall'], 'o-', color='red',
         linewidth=2, label='No Filter', markersize=6)
ax2.plot(epochs, with_filter_metrics['Recall'], 's-', color='blue',
         linewidth=2, label='With Filter (12Hz)', markersize=6)
ax2.set_xlabel('Epoch', fontweight='bold', fontsize=11)
ax2.set_ylabel('Recall', fontweight='bold', fontsize=11)
ax2.set_title('Recall over Epochs', fontweight='bold', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

ax3 = plt.subplot(2, 3, 3)
ax3.plot(epochs, no_filter_metrics['Precision'], 'o-', color='red',
         linewidth=2, label='No Filter', markersize=6)
ax3.plot(epochs, with_filter_metrics['Precision'], 's-', color='blue',
         linewidth=2, label='With Filter (12Hz)', markersize=6)
ax3.set_xlabel('Epoch', fontweight='bold', fontsize=11)
ax3.set_ylabel('Precision', fontweight='bold', fontsize=11)
ax3.set_title('Precision over Epochs', fontweight='bold', fontsize=12)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# 2. Final metrics bar chart
ax4 = plt.subplot(2, 3, 4)
x = np.arange(3)
width = 0.35

no_filter_values = [
    no_filter_metrics['Accuracy'].iloc[-1],
    no_filter_metrics['Recall'].iloc[-1],
    no_filter_metrics['Precision'].iloc[-1]
]

with_filter_values = [
    with_filter_metrics['Accuracy'].iloc[-1],
    with_filter_metrics['Recall'].iloc[-1],
    with_filter_metrics['Precision'].iloc[-1]
]

bars1 = ax4.bar(x - width/2, no_filter_values, width, label='No Filter',
                color='red', alpha=0.7)
bars2 = ax4.bar(x + width/2, with_filter_values, width, label='With Filter (12Hz)',
                color='blue', alpha=0.7)

ax4.set_ylabel('Score', fontweight='bold', fontsize=11)
ax4.set_title('Final Metrics Comparison', fontweight='bold', fontsize=12)
ax4.set_xticks(x)
ax4.set_xticklabels(['Accuracy', 'Recall', 'Precision'])
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

# 3. Training loss comparison
ax5 = plt.subplot(2, 3, 5)
ax5.plot(epochs, no_filter_losses['Train'], 'o-', color='red',
         linewidth=2, label='No Filter', markersize=6)
ax5.plot(epochs, with_filter_losses['Train'], 's-', color='blue',
         linewidth=2, label='With Filter (12Hz)', markersize=6)
ax5.set_xlabel('Epoch', fontweight='bold', fontsize=11)
ax5.set_ylabel('Training Loss', fontweight='bold', fontsize=11)
ax5.set_title('Training Loss over Epochs', fontweight='bold', fontsize=12)
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.3)

# 4. Test loss comparison
ax6 = plt.subplot(2, 3, 6)
ax6.plot(epochs, no_filter_losses['Test'], 'o-', color='red',
         linewidth=2, label='No Filter', markersize=6)
ax6.plot(epochs, with_filter_losses['Test'], 's-', color='blue',
         linewidth=2, label='With Filter (12Hz)', markersize=6)
ax6.set_xlabel('Epoch', fontweight='bold', fontsize=11)
ax6.set_ylabel('Test Loss', fontweight='bold', fontsize=11)
ax6.set_title('Test Loss over Epochs', fontweight='bold', fontsize=12)
ax6.legend(fontsize=10)
ax6.grid(True, alpha=0.3)

plt.suptitle('Training Comparison: 12Hz Filter vs No Filter',
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()

# Save figure
plot_file = os.path.join(output_dir, 'filter_comparison_12hz.png')
plt.savefig(plot_file, dpi=150, bbox_inches='tight')
print(f'✅ Comparison plot saved to: {plot_file}')

plt.show()

# Print summary
print(f'\n{"#"*80}')
print('SUMMARY')
print(f'{"#"*80}')

final_acc_no_filter = no_filter_metrics['Accuracy'].iloc[-1]
final_acc_with_filter = with_filter_metrics['Accuracy'].iloc[-1]
improvement = (final_acc_with_filter - final_acc_no_filter) * 100

if improvement > 0:
    print(f'✅ 12Hz Filtering IMPROVED accuracy by {improvement:.2f}%')
    print(f'   No Filter:    {final_acc_no_filter:.4f}')
    print(f'   With Filter:  {final_acc_with_filter:.4f}')
elif improvement < 0:
    print(f'❌ 12Hz Filtering DECREASED accuracy by {abs(improvement):.2f}%')
    print(f'   No Filter:    {final_acc_no_filter:.4f}')
    print(f'   With Filter:  {final_acc_with_filter:.4f}')
else:
    print(f'➖ 12Hz Filtering had NO EFFECT on accuracy')
    print(f'   Both:         {final_acc_no_filter:.4f}')

print(f'{"#"*80}\n')
print(f'\n✅ COMPARISON COMPLETE!')
print(f'   Results saved to: {output_dir}')
