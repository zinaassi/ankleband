import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from trainer.models.conv1d_model import Conv1DNet


class PrunedConv1DNet(Conv1DNet):
    """Conv1DNet with structured pruning support for ESP32 deployment.

    Extends the base Conv1DNet model with pruning capabilities including:
    - Structured L2-norm pruning on FC layers
    - Pruning mask management
    - Model size calculation
    - Parameter counting
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.pruning_config = None
        if hasattr(cfg.model, 'pruning'):
            self.pruning_config = cfg.model.pruning

    def apply_structured_pruning(self, amount=None, iteration_size=None):
        """Apply L2-norm structured pruning to FC layers.

        Args:
            amount (float): Total pruning amount (e.g., 0.3 for 30%).
                           Uses config if not provided.
            iteration_size (float): Amount to prune in this iteration (e.g., 0.1 for 10%).
                                   If provided, overrides amount parameter.

        Returns:
            dict: Pruning statistics (neurons_before, neurons_after, amount_pruned)
        """
        if not self.pruning_config or not self.pruning_config.enabled:
            print("Pruning is not enabled in config")
            return {}

        # Determine pruning amount for this iteration
        prune_amount = iteration_size if iteration_size is not None else (
            amount if amount is not None else self.pruning_config.amount
        )

        stats = {}

        # Get FC Layer 1 (the first Linear layer in fc_layers)
        fc1_layer = None
        for module in self.fc_layers.modules():
            if isinstance(module, nn.Linear):
                fc1_layer = module
                break

        if fc1_layer is None:
            print("Warning: No Linear layer found to prune")
            return stats

        # Record initial state
        neurons_before = fc1_layer.out_features

        # Apply structured pruning (L2-norm, dimension 0 = output neurons)
        prune.ln_structured(
            module=fc1_layer,
            name='weight',
            amount=prune_amount,
            n=self.pruning_config.norm,  # L2-norm
            dim=self.pruning_config.dim   # Prune output dimension
        )

        # Calculate how many neurons were actually pruned
        mask = getattr(fc1_layer, 'weight_mask', None)
        if mask is not None:
            # Count non-zero rows in mask (remaining neurons)
            neurons_remaining = (mask.sum(dim=1) > 0).sum().item()
        else:
            neurons_remaining = neurons_before

        stats = {
            'neurons_before': neurons_before,
            'neurons_remaining': neurons_remaining,
            'neurons_pruned': neurons_before - neurons_remaining,
            'amount_requested': prune_amount,
            'actual_prune_rate': (neurons_before - neurons_remaining) / neurons_before
        }

        print(f"Pruned FC Layer 1: {neurons_before} → {neurons_remaining} neurons "
              f"({stats['neurons_pruned']} removed, {stats['actual_prune_rate']*100:.1f}%)")

        return stats

    def make_pruning_permanent(self):
        """Remove pruning masks and make pruning permanent.

        This actually deletes the pruned weights from the model,
        reducing the model size. Call this after fine-tuning is complete.

        Returns:
            dict: Size statistics before and after making pruning permanent
        """
        size_before = self.get_model_size()
        params_before = self.count_parameters()

        # Remove pruning masks from all Linear layers
        pruned_count = 0
        for module in self.modules():
            if isinstance(module, nn.Linear):
                try:
                    prune.remove(module, 'weight')
                    pruned_count += 1
                except ValueError:
                    # No pruning mask to remove
                    pass

        size_after = self.get_model_size()
        params_after = self.count_parameters()

        stats = {
            'size_before_kb': size_before,
            'size_after_kb': size_after,
            'size_reduction_kb': size_before - size_after,
            'params_before': params_before,
            'params_after': params_after,
            'layers_made_permanent': pruned_count
        }

        print(f"Made pruning permanent: {size_before:.2f} KB → {size_after:.2f} KB "
              f"({stats['size_reduction_kb']:.2f} KB saved)")
        print(f"Parameters: {params_before} → {params_after} ({params_before - params_after} removed)")

        return stats

    def physically_remove_pruned_neurons(self):
        """Actually resize tensors to remove pruned neurons and reduce model size.

        After calling make_pruning_permanent(), the pruned weights are set to zero
        but the tensors are still the same size. This method creates new, smaller
        layers without the zeroed neurons.

        Must be called AFTER make_pruning_permanent().

        Returns:
            dict: Statistics about removed neurons
        """
        size_before = self.get_model_size()
        params_before = self.count_parameters()

        # Get FC Layer 1 (first Linear in fc_layers)
        fc1 = self.fc_layers[0]
        if not isinstance(fc1, nn.Linear):
            print("Warning: fc_layers[0] is not a Linear layer")
            return {}

        # Get the weight matrix (after mask is applied/removed)
        weights = fc1.weight.data  # Shape: (out_features, in_features) = (64, 200)

        # Find non-zero rows (neurons that weren't fully pruned)
        # A neuron is considered pruned if ALL its weights are zero
        row_norms = weights.norm(dim=1)  # L2 norm of each row
        keep_indices = (row_norms > 1e-6).nonzero(as_tuple=True)[0]

        num_original = weights.shape[0]
        num_kept = len(keep_indices)
        num_removed = num_original - num_kept

        if num_removed == 0:
            print("No neurons to remove - all neurons have non-zero weights")
            return {
                'neurons_before': num_original,
                'neurons_after': num_kept,
                'neurons_removed': 0,
                'size_before_kb': size_before,
                'size_after_kb': size_before,
                'size_reduction_kb': 0
            }

        print(f"Physically removing {num_removed} pruned neurons: {num_original} → {num_kept}")

        # Get device of the model
        device = fc1.weight.device

        # Step 1: Create new smaller FC1 layer
        new_fc1 = nn.Linear(fc1.in_features, num_kept, bias=fc1.bias is not None).to(device)
        new_fc1.weight.data = weights[keep_indices]
        if fc1.bias is not None:
            new_fc1.bias.data = fc1.bias.data[keep_indices]

        # Step 2: Create new BatchNorm layer (fc_layers[1])
        bn1 = self.fc_layers[1]
        if isinstance(bn1, nn.BatchNorm1d):
            new_bn1 = nn.BatchNorm1d(num_kept).to(device)
            new_bn1.weight.data = bn1.weight.data[keep_indices]
            new_bn1.bias.data = bn1.bias.data[keep_indices]
            new_bn1.running_mean = bn1.running_mean[keep_indices]
            new_bn1.running_var = bn1.running_var[keep_indices]
            new_bn1.num_batches_tracked = bn1.num_batches_tracked
        else:
            print(f"Warning: fc_layers[1] is not BatchNorm1d, got {type(bn1)}")
            new_bn1 = bn1

        # Step 3: Update final FC layer input (fc_layers[3])
        fc_final = self.fc_layers[3]
        if isinstance(fc_final, nn.Linear):
            new_fc_final = nn.Linear(num_kept, fc_final.out_features,
                                     bias=fc_final.bias is not None).to(device)
            new_fc_final.weight.data = fc_final.weight.data[:, keep_indices]
            if fc_final.bias is not None:
                new_fc_final.bias.data = fc_final.bias.data
        else:
            print(f"Warning: fc_layers[3] is not Linear, got {type(fc_final)}")
            new_fc_final = fc_final

        # Step 4: Replace layers in Sequential
        self.fc_layers[0] = new_fc1
        self.fc_layers[1] = new_bn1
        self.fc_layers[3] = new_fc_final

        size_after = self.get_model_size()
        params_after = self.count_parameters()

        stats = {
            'neurons_before': num_original,
            'neurons_after': num_kept,
            'neurons_removed': num_removed,
            'size_before_kb': size_before,
            'size_after_kb': size_after,
            'size_reduction_kb': size_before - size_after,
            'compression_ratio': size_before / size_after if size_after > 0 else 1.0,
            'params_before': params_before,
            'params_after': params_after
        }

        print(f"Model size: {size_before:.2f} KB → {size_after:.2f} KB "
              f"({stats['size_reduction_kb']:.2f} KB saved, {stats['compression_ratio']:.2f}x compression)")
        print(f"Parameters: {params_before:,} → {params_after:,} ({params_before - params_after:,} removed)")

        return stats

    def get_model_size(self):
        """Calculate model size in KB (including all parameters and buffers).

        Returns:
            float: Model size in kilobytes
        """
        # Calculate size of all parameters (weights + biases)
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())

        # Calculate size of all buffers (e.g., BatchNorm running_mean/var)
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())

        # Convert bytes to kilobytes
        total_size_kb = (param_size + buffer_size) / 1024

        return total_size_kb

    def count_parameters(self):
        """Count total number of trainable parameters.

        Returns:
            int: Total number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_pruning_statistics(self):
        """Get detailed pruning statistics for all layers.

        Returns:
            dict: Statistics including layer-wise pruning info
        """
        stats = {
            'total_params': self.count_parameters(),
            'model_size_kb': self.get_model_size(),
            'layers': []
        }

        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # Check if this layer has been pruned
                has_mask = hasattr(module, 'weight_mask')

                layer_stats = {
                    'name': name,
                    'in_features': module.in_features,
                    'out_features': module.out_features,
                    'total_weights': module.in_features * module.out_features,
                    'has_pruning_mask': has_mask
                }

                if has_mask:
                    mask = module.weight_mask
                    remaining = mask.sum().item()
                    total = mask.numel()
                    layer_stats['weights_remaining'] = remaining
                    layer_stats['weights_pruned'] = total - remaining
                    layer_stats['sparsity'] = (total - remaining) / total

                stats['layers'].append(layer_stats)

        return stats
