#!/usr/bin/env python3
"""
Quick verification script to test config loading and FILTER_ALPHA parsing.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trainer.utils import ConfigManager

def verify_config(config_name):
    """Load and verify a config file."""
    print(f"\n{'='*60}")
    print(f"Testing: {config_name}")
    print('='*60)

    try:
        cfg = ConfigManager(json_name=f'config/bracelet/{config_name}')

        print(f"✓ Config loaded successfully")
        print(f"  Output dir: {cfg.output_dir}")
        print(f"  Random seed: {cfg.random_seed}")

        # Check filter settings
        print(f"\nFilter Configuration:")
        print(f"  APPLY_FILTER: {cfg.data.apply_filter}")
        print(f"  FILTER_TYPE: {cfg.data.filter_type}")

        if cfg.data.filter_type == 'kalman':
            print(f"  FILTER_Q: {cfg.data.filter_q}")
            print(f"  FILTER_R: {cfg.data.filter_r}")
        elif cfg.data.filter_type == 'ema':
            # Check if filter_alpha exists
            if hasattr(cfg.data, 'filter_alpha'):
                print(f"  FILTER_ALPHA: {cfg.data.filter_alpha}")
                print(f"  ✓ FILTER_ALPHA is correctly parsed!")
            else:
                print(f"  ✗ ERROR: FILTER_ALPHA not found in config!")
                return False

        # Check training settings
        print(f"\nTraining Configuration:")
        print(f"  Batch size: {cfg.training.batch_size}")
        print(f"  Epochs: {cfg.training.epochs}")
        print(f"  Learning rate: {cfg.training.learning_rate}")

        # Check LOO settings
        print(f"\nData Configuration:")
        print(f"  Leave subject out: {cfg.data.leave_subject_out}")
        print(f"  Append: {cfg.data.append}")
        print(f"  Classes: {cfg.data.classes}")

        return True

    except Exception as e:
        print(f"✗ ERROR loading config: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "="*60)
    print("CONFIG VERIFICATION TEST")
    print("="*60)

    configs = [
        'cnn_kalman_q0001.json',
        'cnn_kalman_q00001.json',
        'cnn_ema_a03.json'
    ]

    results = {}
    for config in configs:
        results[config] = verify_config(config)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for config, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{config}: {status}")

    all_passed = all(results.values())
    if all_passed:
        print("\n✓ All configs verified successfully!")
        print("\nReady to submit HPC jobs:")
        print("  bash scripts/05_hpc_job_scripts/run_cnn_filter_comparison.sh")
        return 0
    else:
        print("\n✗ Some configs failed verification!")
        return 1

if __name__ == '__main__':
    sys.exit(main())
