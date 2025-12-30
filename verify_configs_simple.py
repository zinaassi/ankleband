#!/usr/bin/env python3
"""
Simple config verification - just checks JSON structure without dependencies.
"""
import json
import os
from pathlib import Path

def verify_config(config_name):
    """Load and verify a config file."""
    print(f"\n{'='*60}")
    print(f"Testing: {config_name}")
    print('='*60)

    config_path = Path('config/bracelet') / config_name
    if not config_path.exists():
        print(f"✗ ERROR: File not found: {config_path}")
        return False

    try:
        with open(config_path, 'r') as f:
            cfg = json.load(f)

        print(f"✓ JSON loaded successfully")
        print(f"  Output dir: {cfg['OUTPUT_DIR']}")
        print(f"  Random seed: {cfg['RANDOM_SEED']}")

        # Check filter settings
        print(f"\nFilter Configuration:")
        print(f"  APPLY_FILTER: {cfg['DATA']['APPLY_FILTER']}")
        print(f"  FILTER_TYPE: {cfg['DATA']['FILTER_TYPE']}")

        if cfg['DATA']['FILTER_TYPE'] == 'kalman':
            if 'FILTER_Q' in cfg['DATA'] and 'FILTER_R' in cfg['DATA']:
                print(f"  FILTER_Q: {cfg['DATA']['FILTER_Q']}")
                print(f"  FILTER_R: {cfg['DATA']['FILTER_R']}")
                print(f"  ✓ Kalman parameters present!")
            else:
                print(f"  ✗ ERROR: Missing FILTER_Q or FILTER_R!")
                return False

        elif cfg['DATA']['FILTER_TYPE'] == 'ema':
            if 'FILTER_ALPHA' in cfg['DATA']:
                print(f"  FILTER_ALPHA: {cfg['DATA']['FILTER_ALPHA']}")
                print(f"  ✓ FILTER_ALPHA present in JSON!")
            else:
                print(f"  ✗ ERROR: FILTER_ALPHA not found in JSON!")
                return False

        # Check training settings
        print(f"\nTraining Configuration:")
        print(f"  Batch size: {cfg['TRAINING']['BATCH_SIZE']}")
        print(f"  Epochs: {cfg['TRAINING']['EPOCHS']}")
        print(f"  Learning rate: {cfg['TRAINING']['LEARNING_RATE']}")

        # Check LOO settings
        print(f"\nData Configuration:")
        print(f"  Leave subject out: {cfg['DATA']['LEAVE_SUBJECT_OUT']}")
        print(f"  Append: {cfg['DATA']['APPEND']}")
        print(f"  Classes: {cfg['DATA']['CLASSES']}")

        # Check required fields
        required_fields = {
            'DATA': ['PATH', 'TRAIN_FILES', 'APPLY_FILTER', 'FILTER_TYPE'],
            'MODEL': ['TYPE', 'NUM_FC_LAYERS'],
            'TRAINING': ['BATCH_SIZE', 'EPOCHS', 'LEARNING_RATE'],
            'SYSTEM': ['GPU']
        }

        missing = []
        for section, fields in required_fields.items():
            if section not in cfg:
                missing.append(f"{section}")
                continue
            for field in fields:
                if field not in cfg[section]:
                    missing.append(f"{section}.{field}")

        if missing:
            print(f"\n✗ Missing required fields: {', '.join(missing)}")
            return False

        return True

    except json.JSONDecodeError as e:
        print(f"✗ ERROR: Invalid JSON: {e}")
        return False
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "="*60)
    print("CONFIG VERIFICATION TEST (Simple)")
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
        print("\nNext steps:")
        print("1. Submit HPC jobs:")
        print("   bash scripts/05_hpc_job_scripts/run_cnn_filter_comparison.sh")
        print("\n2. After completion, run analysis:")
        print("   python scripts/02_filter_analysis/compare_top3_filters.py")
        return 0
    else:
        print("\n✗ Some configs failed verification!")
        return 1

if __name__ == '__main__':
    import sys
    sys.exit(main())
