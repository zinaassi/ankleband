#!/usr/bin/env python3
"""
Generate Filter Comparison Configs - REDO (Original 6 Filters)

Recreates the 6 filter configs from Phase 1, now with ConfigManager fixed.
Baseline is NOT included (we already have valid baseline results).
"""

import json
import os
from pathlib import Path

# Base configuration template
BASE_CONFIG = {
    "MODE": "regular",
    "DATA": {
        "PATH": "data/dataset",
        "TRAIN_FILES": [
            "ID06_seating_all_gestures.h5",
            "ID06_standing_all_gestures.h5",
            "ID09_seating_all_gestures.h5",
            "ID09_standing_all_gestures.h5",
            "ID02_seating_all_gestures.h5",
            "ID02_standing_all_gestures.h5",
            "ID04_seating_all_gestures.h5",
            "ID04_standing_all_gestures.h5",
            "ID08_seating_all_gestures.h5",
            "ID08_standing_all_gestures.h5",
            "ID03_seating_all_gestures.h5",
            "ID03_standing_all_gestures.h5",
            "ID01_seating_all_gestures.h5",
            "ID01_standing_all_gestures.h5",
            "ID05_seating_all_gestures.h5",
            "ID05_standing_all_gestures.h5",
            "ID10_seating_all_gestures.h5",
            "ID10_standing_all_gestures.h5",
            "ID07_seating_all_gestures.h5",
            "ID07_standing_gestures12.h5",
            "ID07_standing_gestures34free.h5"
        ],
        "TEST_FILES": [],
        "APPEND": 60,
        "STEP": 2,
        "STRIDE": 1,
        "CLASSES": 5,
        "LABEL_PERCENTAGE": 0.5,
        "LEAVE_SUBJECT_OUT": 2,  # Will be overridden
        "SHUFFLE": False,
        "SHARE_TRAIN": 0.0,
        "K_FOLD": None,
        "SINGLE_TEST": False,
        "DTW": False,
        "FORCE_NUM_SUBJECTS_TRAIN": None
    },
    "MODEL": {
        "TYPE": "neuralnet",
        "NUM_FC_LAYERS": 2,
        "WEIGHTS": ""
    },
    "TRAINING": {
        "BATCH_SIZE": 64,
        "EPOCHS": 10,
        "CP_INTERVAL": 5,
        "LEARNING_RATE": 0.0001,
        "WEIGHT_DECAY": 5e-05,
        "EPSILON": 1e-08,
        "MOMENTUM": 0.999,
        "SCHEDULER_STEPS": 10000000,
        "GRADIENT_CLIP": 1.0,
        "BATCH_NUM_WORKERS": 2,
        "WEIGHTED_SAMPLING": False
    },
    "SYSTEM": {
        "GPU": 0
    },
    "RANDOM_SEED": 42,
    "STORE_CSV": False
}

# Filter configurations (matching your original Phase 1 experiments)
FILTER_CONFIGS = [
    # Butterworth 40Hz Order 2
    {
        'name': 'butterworth_40hz_o2',
        'display_name': 'Butterworth (40Hz, O2)',
        'apply_filter': True,
        'filter_type': 'butterworth',
        'filter_cutoff': 40,
        'filter_order': 2,
    },
    # Biquad 30Hz Q=1.0
    {
        'name': 'biquad_30hz_q10',
        'display_name': 'Biquad (30Hz, Q=1.0)',
        'apply_filter': True,
        'filter_type': 'biquad',
        'filter_cutoff': 30,
        'filter_q': 1.0,
    },
    # EMA alpha=0.5
    {
        'name': 'ema_alpha05',
        'display_name': 'EMA (α=0.5)',
        'apply_filter': True,
        'filter_type': 'ema',
        'filter_alpha': 0.5,
    },
    # Kalman Q=0.0001 R=0.0001
    {
        'name': 'kalman_q0001_r0001',
        'display_name': 'Kalman (Q=0.0001, R=0.0001)',
        'apply_filter': True,
        'filter_type': 'kalman',
        'filter_q': 0.0001,
        'filter_r': 0.0001,
    },
    # Kalman Light Q=0.1 R=0.1
    {
        'name': 'kalman_light_q01_r01',
        'display_name': 'Kalman Light (Q=0.1, R=0.1)',
        'apply_filter': True,
        'filter_type': 'kalman',
        'filter_q': 0.1,
        'filter_r': 0.1,
    },
    # Kalman Smooth Q=0.001 R=0.1
    {
        'name': 'kalman_smooth_q001_r01',
        'display_name': 'Kalman Smooth (Q=0.001, R=0.1)',
        'apply_filter': True,
        'filter_type': 'kalman',
        'filter_q': 0.001,
        'filter_r': 0.1,
    },
]

SUBJECTS = [2, 3, 6]


def create_config(filter_config, subject):
    """Create a single config file."""
    config = json.loads(json.dumps(BASE_CONFIG))  # Deep copy
    
    # Set subject
    config['DATA']['LEAVE_SUBJECT_OUT'] = subject
    
    # Set filter parameters
    config['DATA']['APPLY_FILTER'] = filter_config['apply_filter']
    config['DATA']['FILTER_TYPE'] = filter_config['filter_type']
    
    if 'filter_cutoff' in filter_config:
        config['DATA']['FILTER_CUTOFF'] = filter_config['filter_cutoff']
    if 'filter_order' in filter_config:
        config['DATA']['FILTER_ORDER'] = filter_config['filter_order']
    if 'filter_q' in filter_config:
        config['DATA']['FILTER_Q'] = filter_config['filter_q']
    if 'filter_r' in filter_config:
        config['DATA']['FILTER_R'] = filter_config['filter_r']
    if 'filter_alpha' in filter_config:
        config['DATA']['FILTER_ALPHA'] = filter_config['filter_alpha']
    
    # Set output directory
    config['OUTPUT_DIR'] = f"outputs/filter_redo/{filter_config['name']}_s{subject:02d}"
    
    return config


def main():
    """Generate all config files."""
    # Create output directory
    config_dir = Path('config/filter_redo')
    config_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("GENERATING FILTER COMPARISON CONFIGS (REDO - Original 6 Filters)")
    print("=" * 80)
    print("\nNOTE: Baseline NOT included - using existing baseline results")
    print()
    
    configs_generated = []
    
    # Account 1: Butterworth, Biquad, EMA
    account1_filters = ['butterworth_40hz_o2', 'biquad_30hz_q10', 'ema_alpha05']
    account1_configs = []
    
    # Account 2: Kalman variants
    account2_filters = ['kalman_q0001_r0001', 'kalman_light_q01_r01', 'kalman_smooth_q001_r01']
    account2_configs = []
    
    for filter_cfg in FILTER_CONFIGS:
        for subject in SUBJECTS:
            config = create_config(filter_cfg, subject)
            filename = f"{filter_cfg['name']}_s{subject:02d}.json"
            filepath = config_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=2)
            
            configs_generated.append((filename, filter_cfg['display_name']))
            
            # Categorize by account
            if filter_cfg['name'] in account1_filters:
                account1_configs.append((filename, filter_cfg['display_name']))
            else:
                account2_configs.append((filename, filter_cfg['display_name']))
    
    print(f"✓ Generated {len(configs_generated)} config files in config/filter_redo/\n")
    
    # Print split with display names
    print("=" * 80)
    print("EXPERIMENT SPLIT")
    print("=" * 80)
    
    print(f"\n**Account 1 (badran.abed)** - {len(account1_configs)} experiments:")
    current_filter = None
    for cfg, display in account1_configs:
        if display != current_filter:
            print(f"\n  {display}:")
            current_filter = display
        print(f"    - {cfg}")
    
    print(f"\n**Account 2 (zina.assi)** - {len(account2_configs)} experiments:")
    current_filter = None
    for cfg, display in account2_configs:
        if display != current_filter:
            print(f"\n  {display}:")
            current_filter = display
        print(f"    - {cfg}")
    
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("\n1. Fix trainer/utils.py ConfigDataManager (add filter parsing)")
    print("\n2. Test ONE config to verify fix:")
    print("   python trainer/train_conv.py --json config/filter_redo/butterworth_40hz_o2_s02.json --loo 2")
    print("\n3. Run experiments on both accounts:")
    print("   bash scripts/filter_redo/run_account1_filters.sh")
    print("   bash scripts/filter_redo/run_account2_filters.sh")
    print("\n4. Compare with original (incorrect) results to see real differences!")
    print()


if __name__ == '__main__':
    main()
