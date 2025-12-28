#!/usr/bin/env python3
"""
Structured Pruning with Fine-tuning for Conv1DNet

Implements hybrid iterative pruning with LR rewinding for ESP32 deployment:
- 10% target: 2 iterations of 5% (gentler)
- 20-50% targets: Multiple iterations of 10% each

Usage:
    python prune_and_finetune.py --json config/pruning/prune_kalman_s02_30pct_seed42.json
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, recall_score, precision_score

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from trainer.utils import ConfigManager
from trainer.models.pruned_conv1d_model import PrunedConv1DNet
from data.load_data import DataManagement, TorchDatasetManagement


def get_pruning_schedule(target_amount):
    """Determine pruning iterations based on target amount (hybrid approach).

    Args:
        target_amount (float): Total target pruning (e.g., 0.3 for 30%)

    Returns:
        list: List of (iteration_size, num_epochs, learning_rates) tuples
    """
    schedules = {
        0.1: [  # 10% total: 2 iterations of 5%
            (0.05, 3, [1e-4, 1e-5, 1e-5]),
            (0.05, 3, [1e-4, 1e-5, 1e-5]),
        ],
        0.2: [  # 20% total: 2 iterations of 10%
            (0.10, 3, [1e-4, 1e-5, 1e-5]),
            (0.10, 4, [1e-4, 1e-5, 1e-5, 1e-6]),
        ],
        0.3: [  # 30% total: 3 iterations of 10%
            (0.10, 3, [1e-4, 1e-5, 1e-5]),
            (0.10, 3, [1e-4, 1e-5, 1e-5]),
            (0.10, 4, [1e-5, 1e-5, 1e-6, 1e-6]),
        ],
        0.4: [  # 40% total: 4 iterations of 10%
            (0.10, 3, [1e-4, 1e-5, 1e-5]),
            (0.10, 3, [1e-5, 1e-5, 1e-6]),
            (0.10, 3, [1e-5, 1e-5, 1e-6]),
            (0.10, 4, [1e-5, 1e-6, 1e-6, 1e-6]),
        ],
        0.5: [  # 50% total: 5 iterations of 10%
            (0.10, 3, [1e-4, 1e-5, 1e-5]),
            (0.10, 3, [1e-5, 1e-5, 1e-6]),
            (0.10, 3, [1e-5, 1e-6, 1e-6]),
            (0.10, 3, [1e-6, 1e-6, 1e-6]),
            (0.10, 4, [1e-6, 1e-6, 1e-6, 1e-6]),
        ],
    }

    # Round to nearest 10%
    rounded_amount = round(target_amount * 10) / 10

    if rounded_amount not in schedules:
        raise ValueError(f"Unsupported pruning amount: {target_amount}. "
                        f"Supported: {list(schedules.keys())}")

    return schedules[rounded_amount]


def evaluate_model(model, test_loader, device, cfg):
    """Evaluate model and return metrics.

    Args:
        model: Model to evaluate
        test_loader: Test data loader
        device: Device to run on
        cfg: Configuration

    Returns:
        dict: Metrics (accuracy, recall, precision, loss)
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    all_preds = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device).float()
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total_loss += loss.item()
            num_batches += 1

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    return {
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision,
        'loss': avg_loss
    }


def fine_tune_iteration(model, train_loader, test_loader, device, cfg,
                        num_epochs, learning_rates):
    """Fine-tune model for one pruning iteration.

    Args:
        model: Model to fine-tune
        train_loader: Training data loader
        test_loader: Test data loader
        device: Device to run on
        cfg: Configuration
        num_epochs: Number of epochs for this iteration
        learning_rates: List of learning rates (one per epoch)

    Returns:
        tuple: (train_metrics_list, test_metrics_list)
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rates[0],
        weight_decay=cfg.training.weight_decay,
        eps=cfg.training.epsilon
    )

    train_metrics_list = []
    test_metrics_list = []

    for epoch in range(num_epochs):
        # Set learning rate for this epoch
        if epoch < len(learning_rates):
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rates[epoch]

        # Training phase
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device).float()
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping
            if cfg.training.gradient_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip)

            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_train_loss = epoch_loss / num_batches if num_batches > 0 else 0.0

        # Evaluation phase
        test_metrics = evaluate_model(model, test_loader, device, cfg)
        test_metrics['epoch'] = epoch + 1
        test_metrics['lr'] = learning_rates[epoch] if epoch < len(learning_rates) else learning_rates[-1]

        train_metrics_list.append({'epoch': epoch + 1, 'loss': avg_train_loss})
        test_metrics_list.append(test_metrics)

        print(f"  Epoch {epoch + 1}/{num_epochs}: "
              f"Train Loss = {avg_train_loss:.4f}, "
              f"Test Acc = {test_metrics['accuracy']:.4f}, "
              f"LR = {test_metrics['lr']:.2e}")

    return train_metrics_list, test_metrics_list


def main():
    parser = argparse.ArgumentParser(description='Prune and fine-tune Conv1DNet')
    parser.add_argument('--json', '-json', type=str, required=True,
                       help='Path to JSON config file')
    args = parser.parse_args()

    # Load configuration
    print(f"Loading config from: {args.json}")
    cfg = ConfigManager(json_name=args.json)

    # Create output directory
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Save config to output directory
    import shutil
    shutil.copy(args.json, os.path.join(cfg.output_dir, 'config.json'))

    # Set device
    device = torch.device(f'cuda:{cfg.system.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set random seed
    torch.manual_seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)

    # Load data
    print("Loading data...")
    data_manager = DataManagement(cfg=cfg)
    train_dataset = TorchDatasetManagement(cfg=cfg, data_df=data_manager.train_df, inputs_names_stacked=data_manager.inputs_names_stacked, is_train=True)
    test_dataset = TorchDatasetManagement(cfg=cfg, data_df=data_manager.test_df, inputs_names_stacked=data_manager.inputs_names_stacked, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.batch_num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.batch_num_workers
    )

    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    # Load pretrained model
    print(f"Loading pretrained model from: {cfg.model.weights}")
    model = PrunedConv1DNet(cfg=cfg)
    model.load_state_dict(torch.load(cfg.model.weights, map_location=device))
    model.to(device).float()

    # Evaluate baseline (before pruning)
    print("\n" + "="*80)
    print("BASELINE EVALUATION (Before Pruning)")
    print("="*80)
    baseline_metrics = evaluate_model(model, test_loader, device, cfg)
    baseline_size = model.get_model_size()
    baseline_params = model.count_parameters()

    print(f"Baseline Accuracy: {baseline_metrics['accuracy']:.4f}")
    print(f"Baseline Recall:   {baseline_metrics['recall']:.4f}")
    print(f"Baseline Precision: {baseline_metrics['precision']:.4f}")
    print(f"Baseline Size:     {baseline_size:.2f} KB")
    print(f"Baseline Params:   {baseline_params:,}")

    # Get pruning schedule
    target_amount = cfg.model.pruning.amount
    pruning_schedule = get_pruning_schedule(target_amount)

    print(f"\nTarget Pruning: {target_amount*100:.0f}%")
    print(f"Pruning Schedule: {len(pruning_schedule)} iterations")

    # Track all metrics
    all_train_metrics = []
    all_test_metrics = []
    iteration_summaries = []

    # Iterative pruning + fine-tuning
    cumulative_epochs = 0
    for iteration, (prune_size, num_epochs, lrs) in enumerate(pruning_schedule, 1):
        print(f"\n" + "="*80)
        print(f"ITERATION {iteration}/{len(pruning_schedule)}: Prune {prune_size*100:.0f}%, Fine-tune {num_epochs} epochs")
        print("="*80)

        # Apply pruning
        print(f"\nApplying pruning...")
        prune_stats = model.apply_structured_pruning(iteration_size=prune_size)

        # Evaluate after pruning (before fine-tuning)
        print(f"\nEvaluating after pruning (before fine-tuning)...")
        after_prune_metrics = evaluate_model(model, test_loader, device, cfg)
        print(f"After Pruning: Accuracy = {after_prune_metrics['accuracy']:.4f} "
              f"(drop: {baseline_metrics['accuracy'] - after_prune_metrics['accuracy']:.4f})")

        # Fine-tune
        print(f"\nFine-tuning for {num_epochs} epochs with LRs: {lrs}...")
        train_metrics, test_metrics = fine_tune_iteration(
            model, train_loader, test_loader, device, cfg,
            num_epochs, lrs
        )

        # Update cumulative epoch count
        for tm in train_metrics:
            tm['cumulative_epoch'] = cumulative_epochs + tm['epoch']
            tm['iteration'] = iteration
        for tm in test_metrics:
            tm['cumulative_epoch'] = cumulative_epochs + tm['epoch']
            tm['iteration'] = iteration

        cumulative_epochs += num_epochs
        all_train_metrics.extend(train_metrics)
        all_test_metrics.extend(test_metrics)

        # Final evaluation for this iteration
        final_metrics = test_metrics[-1]  # Last epoch
        current_size = model.get_model_size()
        current_params = model.count_parameters()

        iteration_summary = {
            'iteration': iteration,
            'prune_amount': prune_size,
            'neurons_before': prune_stats.get('neurons_before', 0),
            'neurons_after': prune_stats.get('neurons_remaining', 0),
            'accuracy': final_metrics['accuracy'],
            'recall': final_metrics['recall'],
            'precision': final_metrics['precision'],
            'model_size_kb': current_size,
            'parameters': current_params
        }
        iteration_summaries.append(iteration_summary)

        print(f"\nIteration {iteration} Summary:")
        print(f"  Accuracy:   {final_metrics['accuracy']:.4f}")
        print(f"  Recall:     {final_metrics['recall']:.4f}")
        print(f"  Precision:  {final_metrics['precision']:.4f}")
        print(f"  Model Size: {current_size:.2f} KB")
        print(f"  Parameters: {current_params:,}")

        # Save intermediate model
        intermediate_path = os.path.join(cfg.output_dir, f'pruned_iter{iteration}.pt')
        torch.save(model.state_dict(), intermediate_path)
        print(f"  Saved: {intermediate_path}")

    # Make pruning permanent
    print(f"\n" + "="*80)
    print("FINALIZING: Making pruning permanent")
    print("="*80)
    permanent_stats = model.make_pruning_permanent()

    # Final evaluation
    print(f"\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80)
    final_metrics = evaluate_model(model, test_loader, device, cfg)
    final_size = model.get_model_size()
    final_params = model.count_parameters()

    print(f"Final Accuracy:   {final_metrics['accuracy']:.4f} (baseline: {baseline_metrics['accuracy']:.4f})")
    print(f"Accuracy Drop:    {baseline_metrics['accuracy'] - final_metrics['accuracy']:.4f}")
    print(f"Final Recall:     {final_metrics['recall']:.4f}")
    print(f"Final Precision:  {final_metrics['precision']:.4f}")
    print(f"Final Size:       {final_size:.2f} KB (baseline: {baseline_size:.2f} KB)")
    print(f"Size Reduction:   {baseline_size - final_size:.2f} KB ({(baseline_size - final_size)/baseline_size*100:.1f}%)")
    print(f"Final Parameters: {final_params:,} (baseline: {baseline_params:,})")
    print(f"Compression Ratio: {baseline_size / final_size:.2f}×")

    # Save final model
    final_path = os.path.join(cfg.output_dir, 'pruned_final.pt')
    torch.save(model.state_dict(), final_path)
    print(f"\nSaved final model: {final_path}")

    # Save metrics to CSV
    train_df = pd.DataFrame(all_train_metrics)
    test_df = pd.DataFrame(all_test_metrics)
    iteration_df = pd.DataFrame(iteration_summaries)

    train_df.to_csv(os.path.join(cfg.output_dir, 'train_losses.csv'), index=False)
    test_df.to_csv(os.path.join(cfg.output_dir, 'test_metrics.csv'), index=False)
    iteration_df.to_csv(os.path.join(cfg.output_dir, 'iteration_summary.csv'), index=False)

    # Save final summary
    summary = {
        'baseline_accuracy': baseline_metrics['accuracy'],
        'baseline_recall': baseline_metrics['recall'],
        'baseline_precision': baseline_metrics['precision'],
        'baseline_size_kb': baseline_size,
        'baseline_params': baseline_params,
        'final_accuracy': final_metrics['accuracy'],
        'final_recall': final_metrics['recall'],
        'final_precision': final_metrics['precision'],
        'final_size_kb': final_size,
        'final_params': final_params,
        'accuracy_drop': baseline_metrics['accuracy'] - final_metrics['accuracy'],
        'size_reduction_kb': baseline_size - final_size,
        'compression_ratio': baseline_size / final_size,
        'target_pruning_pct': target_amount * 100,
        'num_iterations': len(pruning_schedule),
        'total_epochs': cumulative_epochs
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join(cfg.output_dir, 'final_summary.csv'), index=False)

    print(f"\nAll results saved to: {cfg.output_dir}")
    print("="*80)
    print("PRUNING COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
