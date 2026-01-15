#!/usr/bin/env python3
"""
INT8 Weight Export Script for Quantized Models

Exports INT8 weights from a PyTorch quantized model state_dict to C++ header format.
Extracts:
- INT8 weights (int8_t arrays)
- Scale and zero_point for each quantized layer
- BatchNorm parameters (kept as FP32)

Usage:
    python model_conversions/export_int8_weights.py \
        --model outputs/quantization/qat_s03_seed42/quantized_model.pt \
        --output rt_code/model_weights_int8_s03.h
"""

import argparse
import os
import sys
import numpy as np

# Add project root to path
sys.path.insert(0, '.')
sys.path.insert(0, '..')

import torch
import torch.nn as nn


def extract_weights_from_state_dict(state_dict):
    """
    Extract INT8 weights and quantization parameters from a quantized model state_dict.

    Handles:
    - Quantized Conv1d weights (stored as regular tensors with scale/zero_point)
    - Quantized Linear weights (stored as _packed_params)
    - BatchNorm parameters (FP32)

    Returns:
        dict: Dictionary containing weights, scales, zero_points for each layer
    """
    weights = {}

    print("\nExtracting weights from state_dict...")
    print("=" * 60)

    # Extract Conv1D weights
    if 'conv1d_1.weight' in state_dict:
        conv_weight = state_dict['conv1d_1.weight']
        conv_scale = state_dict.get('conv1d_1.scale', torch.tensor(1.0))
        conv_zp = state_dict.get('conv1d_1.zero_point', torch.tensor(0))

        # Check if it's a quantized tensor
        if hasattr(conv_weight, 'int_repr'):
            int8_weights = conv_weight.int_repr().numpy()
            scale = conv_weight.q_scale()
            zero_point = conv_weight.q_zero_point()
        else:
            # It's stored as FP32, quantize it using the scale/zp
            scale = conv_scale.item() if conv_scale.numel() == 1 else 0.01
            zero_point = int(conv_zp.item()) if conv_zp.numel() == 1 else 0
            # Quantize: q = round(x / scale) + zp
            int8_weights = np.clip(
                np.round(conv_weight.numpy() / scale) + zero_point,
                -128, 127
            ).astype(np.int8)

        weights['conv1d_1_weight'] = {
            'int8': int8_weights,
            'scale': float(scale),
            'zero_point': int(zero_point),
            'shape': int8_weights.shape,
            'type': 'quantized'
        }
        print(f"  conv1d_1: weight shape {int8_weights.shape}, scale={scale:.6f}, zp={zero_point}")

    # Extract Conv1D bias if present
    if 'conv1d_1.bias' in state_dict:
        bias = state_dict['conv1d_1.bias'].detach().numpy()
        weights['conv1d_1_bias'] = {
            'fp32': bias,
            'shape': bias.shape,
            'type': 'fp32'
        }
        print(f"  conv1d_1: bias shape {bias.shape} (FP32)")

    # Extract BatchNorm1 parameters
    if 'bn1.weight' in state_dict:
        weights['bn1_weight'] = {'fp32': state_dict['bn1.weight'].numpy(), 'type': 'fp32'}
        weights['bn1_bias'] = {'fp32': state_dict['bn1.bias'].numpy(), 'type': 'fp32'}
        weights['bn1_running_mean'] = {'fp32': state_dict['bn1.running_mean'].numpy(), 'type': 'fp32'}
        weights['bn1_running_var'] = {'fp32': state_dict['bn1.running_var'].numpy(), 'type': 'fp32'}
        print(f"  bn1: size {state_dict['bn1.weight'].shape[0]} (FP32)")

    # Extract FC1 (packed params format)
    if 'fc_layers.0._packed_params._packed_params' in state_dict:
        packed = state_dict['fc_layers.0._packed_params._packed_params']
        fc1_scale = state_dict.get('fc_layers.0.scale', torch.tensor(0.01))
        fc1_zp = state_dict.get('fc_layers.0.zero_point', torch.tensor(0))

        # Unpack: (weight, bias)
        if isinstance(packed, tuple):
            weight_packed, bias = packed

            # For quantized linear, weight is stored as a quantized tensor
            if hasattr(weight_packed, 'int_repr'):
                int8_weights = weight_packed.int_repr().numpy()
                scale = weight_packed.q_scale()
                zero_point = weight_packed.q_zero_point()
            else:
                # Fallback: use the scale/zp from state_dict
                scale = fc1_scale.item() if fc1_scale.numel() == 1 else 0.01
                zero_point = int(fc1_zp.item()) if fc1_zp.numel() == 1 else 0
                int8_weights = np.clip(
                    np.round(weight_packed.numpy() / scale) + zero_point,
                    -128, 127
                ).astype(np.int8)

            weights['fc_layers_0_weight'] = {
                'int8': int8_weights,
                'scale': float(scale),
                'zero_point': int(zero_point),
                'shape': int8_weights.shape,
                'type': 'quantized'
            }
            print(f"  fc_layers.0: weight shape {int8_weights.shape}, scale={scale:.6f}, zp={zero_point}")

            # Bias is FP32
            if bias is not None:
                weights['fc_layers_0_bias'] = {
                    'fp32': bias.detach().numpy(),
                    'shape': bias.shape,
                    'type': 'fp32'
                }
                print(f"  fc_layers.0: bias shape {bias.shape} (FP32)")

    # Extract BatchNorm2 parameters (fc_layers.1)
    if 'fc_layers.1.weight' in state_dict:
        weights['bn2_weight'] = {'fp32': state_dict['fc_layers.1.weight'].numpy(), 'type': 'fp32'}
        weights['bn2_bias'] = {'fp32': state_dict['fc_layers.1.bias'].numpy(), 'type': 'fp32'}
        weights['bn2_running_mean'] = {'fp32': state_dict['fc_layers.1.running_mean'].numpy(), 'type': 'fp32'}
        weights['bn2_running_var'] = {'fp32': state_dict['fc_layers.1.running_var'].numpy(), 'type': 'fp32'}
        print(f"  bn2 (fc_layers.1): size {state_dict['fc_layers.1.weight'].shape[0]} (FP32)")

    # Extract FC2 (packed params format) - fc_layers.3
    if 'fc_layers.3._packed_params._packed_params' in state_dict:
        packed = state_dict['fc_layers.3._packed_params._packed_params']
        fc2_scale = state_dict.get('fc_layers.3.scale', torch.tensor(0.01))
        fc2_zp = state_dict.get('fc_layers.3.zero_point', torch.tensor(0))

        if isinstance(packed, tuple):
            weight_packed, bias = packed

            if hasattr(weight_packed, 'int_repr'):
                int8_weights = weight_packed.int_repr().numpy()
                scale = weight_packed.q_scale()
                zero_point = weight_packed.q_zero_point()
            else:
                scale = fc2_scale.item() if fc2_scale.numel() == 1 else 0.01
                zero_point = int(fc2_zp.item()) if fc2_zp.numel() == 1 else 0
                int8_weights = np.clip(
                    np.round(weight_packed.numpy() / scale) + zero_point,
                    -128, 127
                ).astype(np.int8)

            weights['fc_layers_3_weight'] = {
                'int8': int8_weights,
                'scale': float(scale),
                'zero_point': int(zero_point),
                'shape': int8_weights.shape,
                'type': 'quantized'
            }
            print(f"  fc_layers.3: weight shape {int8_weights.shape}, scale={scale:.6f}, zp={zero_point}")

            if bias is not None:
                weights['fc_layers_3_bias'] = {
                    'fp32': bias.detach().numpy(),
                    'shape': bias.shape,
                    'type': 'fp32'
                }
                print(f"  fc_layers.3: bias shape {bias.shape} (FP32)")

    return weights


def write_int8_header(weights, output_path, model_name="s03"):
    """
    Write weights to C++ header file in INT8 format.

    Args:
        weights: Dictionary of extracted weights
        output_path: Path to output header file
        model_name: Name suffix for the model
    """
    with open(output_path, 'w') as f:
        # Header
        f.write(f'// model_weights_int8_{model_name}.h\n')
        f.write('// INT8 quantized weights for optimized neural network\n')
        f.write('// Auto-generated by export_int8_weights.py\n')
        f.write('\n')
        f.write('#ifndef MODEL_WEIGHTS_INT8_H\n')
        f.write('#define MODEL_WEIGHTS_INT8_H\n')
        f.write('\n')
        f.write('#include <stdint.h>\n')
        f.write('\n')

        # Track total sizes
        total_int8_bytes = 0
        total_fp32_bytes = 0

        # Write each weight array
        for name, data in weights.items():
            if data['type'] == 'quantized':
                # INT8 weights
                arr = data['int8']
                scale = data['scale']
                zero_point = data['zero_point']

                # Write scale and zero_point
                safe_name = name.replace('.', '_')
                f.write(f'// {safe_name}: scale={scale:.8f}, zero_point={zero_point}\n')
                f.write(f'const float {safe_name}_scale = {scale:.8f}f;\n')
                f.write(f'const int8_t {safe_name}_zero_point = {zero_point};\n')

                # Write INT8 array
                if len(arr.shape) == 3:
                    # Conv1D weights: [out_channels, in_channels, kernel_size]
                    f.write(f'const int {safe_name}_num_kernels = {arr.shape[0]};\n')
                    f.write(f'const int {safe_name}_num_channels = {arr.shape[1]};\n')
                    f.write(f'const int {safe_name}_kernel_size = {arr.shape[2]};\n')
                    f.write(f'const int8_t {safe_name}[{arr.shape[0]}][{arr.shape[1]}][{arr.shape[2]}] = {{\n')
                    for i in range(arr.shape[0]):
                        f.write('    {\n')
                        for j in range(arr.shape[1]):
                            f.write('        {')
                            f.write(', '.join(str(int(x)) for x in arr[i, j]))
                            f.write('}')
                            if j < arr.shape[1] - 1:
                                f.write(',')
                            f.write('\n')
                        f.write('    }')
                        if i < arr.shape[0] - 1:
                            f.write(',')
                        f.write('\n')
                    f.write('};\n\n')
                    total_int8_bytes += arr.size

                elif len(arr.shape) == 2:
                    # Linear weights: [out_features, in_features]
                    f.write(f'const int {safe_name}_num_rows = {arr.shape[0]};\n')
                    f.write(f'const int {safe_name}_num_cols = {arr.shape[1]};\n')
                    f.write(f'const int8_t {safe_name}[{arr.shape[0]}][{arr.shape[1]}] = {{\n')
                    for i in range(arr.shape[0]):
                        f.write('    {')
                        # Split into chunks to avoid very long lines
                        row_vals = [str(int(x)) for x in arr[i]]
                        f.write(', '.join(row_vals))
                        f.write('}')
                        if i < arr.shape[0] - 1:
                            f.write(',')
                        f.write('\n')
                    f.write('};\n\n')
                    total_int8_bytes += arr.size

                elif len(arr.shape) == 1:
                    # Bias
                    f.write(f'const int {safe_name}_size = {arr.shape[0]};\n')
                    f.write(f'const int8_t {safe_name}[{arr.shape[0]}] = {{\n    ')
                    f.write(', '.join(str(int(x)) for x in arr))
                    f.write('\n};\n\n')
                    total_int8_bytes += arr.size

            else:
                # FP32 weights (BatchNorm, biases, etc.)
                arr = data['fp32']
                safe_name = name.replace('.', '_')

                if len(arr.shape) == 1:
                    f.write(f'const int {safe_name}_size = {arr.shape[0]};\n')
                    f.write(f'const float {safe_name}[{arr.shape[0]}] = {{\n    ')
                    f.write(', '.join(f'{x:.6f}f' for x in arr))
                    f.write('\n};\n\n')
                    total_fp32_bytes += arr.size * 4

                elif len(arr.shape) == 2:
                    f.write(f'const int {safe_name}_num_rows = {arr.shape[0]};\n')
                    f.write(f'const int {safe_name}_num_cols = {arr.shape[1]};\n')
                    f.write(f'const float {safe_name}[{arr.shape[0]}][{arr.shape[1]}] = {{\n')
                    for i in range(arr.shape[0]):
                        f.write('    {')
                        f.write(', '.join(f'{x:.6f}f' for x in arr[i]))
                        f.write('}')
                        if i < arr.shape[0] - 1:
                            f.write(',')
                        f.write('\n')
                    f.write('};\n\n')
                    total_fp32_bytes += arr.size * 4

        # Footer with size summary
        f.write('// Memory Summary:\n')
        f.write(f'// INT8 weights: {total_int8_bytes} bytes ({total_int8_bytes/1024:.2f} KB)\n')
        f.write(f'// FP32 params:  {total_fp32_bytes} bytes ({total_fp32_bytes/1024:.2f} KB)\n')
        f.write(f'// Total:        {total_int8_bytes + total_fp32_bytes} bytes ({(total_int8_bytes + total_fp32_bytes)/1024:.2f} KB)\n')
        f.write('\n')
        f.write('#endif // MODEL_WEIGHTS_INT8_H\n')

    print(f"\nMemory Summary:")
    print(f"  INT8 weights: {total_int8_bytes} bytes ({total_int8_bytes/1024:.2f} KB)")
    print(f"  FP32 params:  {total_fp32_bytes} bytes ({total_fp32_bytes/1024:.2f} KB)")
    print(f"  Total:        {total_int8_bytes + total_fp32_bytes} bytes ({(total_int8_bytes + total_fp32_bytes)/1024:.2f} KB)")


def main():
    parser = argparse.ArgumentParser(description='Export INT8 weights from quantized model')
    parser.add_argument('--model', '-m', required=True,
                        help='Path to quantized model (.pt file)')
    parser.add_argument('--output', '-o', required=True,
                        help='Output path for C++ header file')
    parser.add_argument('--name', '-n', default='s03',
                        help='Model name suffix (default: s03)')
    args = parser.parse_args()

    print(f"Loading quantized model from: {args.model}")

    # Load state_dict
    state_dict = torch.load(args.model, map_location='cpu')

    # If it's a full model, get state_dict
    if not isinstance(state_dict, dict):
        print("Model is a full object, extracting state_dict...")
        state_dict = state_dict.state_dict()

    print(f"Loaded state_dict with {len(state_dict)} keys")

    print("\n" + "=" * 60)
    print("Extracting weights...")
    print("=" * 60)

    # Extract weights
    weights = extract_weights_from_state_dict(state_dict)

    print("\n" + "=" * 60)
    print(f"Writing to: {args.output}")
    print("=" * 60)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)

    # Write header file
    write_int8_header(weights, args.output, args.name)

    print(f"\nDone! Header file written to: {args.output}")


if __name__ == '__main__':
    main()
