#!/usr/bin/env python3
"""
Helper function to evaluate models (pruned FP32 or quantized INT8).
Calculates accuracy, recall, precision, F1 score.
"""

import torch
import numpy as np


def evaluate_model(model, data_loader, device, num_classes=5):
    """
    Evaluate a model (FP32 or INT8) on a dataset.

    Args:
        model: The model to evaluate (can be FP32 or quantized)
        data_loader: DataLoader with test/validation data
        device: torch.device or 'cpu'
        num_classes: Number of output classes

    Returns:
        dict: Metrics (accuracy, recall, precision, f1)
    """
    model.eval()

    # Ensure model is on correct device
    if str(device) != 'cpu':
        model = model.to(device)
    else:
        model = model.to('cpu')

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            # Move to device
            if str(device) != 'cpu':
                inputs = inputs.to(device).float()
                targets = targets.to(device).long()
            else:
                inputs = inputs.to('cpu').float()
                targets = targets.to('cpu').long()

            # Forward pass
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Calculate metrics
    accuracy = (all_preds == all_targets).mean()

    # Per-class recall and precision
    recall_per_class = []
    precision_per_class = []

    for cls in range(num_classes):
        tp = ((all_preds == cls) & (all_targets == cls)).sum()
        fn = ((all_preds != cls) & (all_targets == cls)).sum()
        fp = ((all_preds == cls) & (all_targets != cls)).sum()

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        recall_per_class.append(recall)
        precision_per_class.append(precision)

    avg_recall = np.mean(recall_per_class)
    avg_precision = np.mean(precision_per_class)

    # F1 score
    if avg_precision + avg_recall > 0:
        f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
    else:
        f1_score = 0.0

    return {
        'accuracy': accuracy,
        'recall': avg_recall,
        'precision': avg_precision,
        'f1': f1_score
    }
