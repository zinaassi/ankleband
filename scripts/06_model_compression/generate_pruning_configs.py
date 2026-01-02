#!/usr/bin/env python3
"""
Generate Pruning Configuration Files

Creates 45 JSON configuration files for the pruning experiment sweep:
- 3 subjects (2, 3, 6)
- 5 pruning levels (10%, 20%, 30%, 40%, 50%)
- 3 random seeds (42, 123, 456)

Usage:
    python generate_pruning_configs.py
"""

import json
import os
from pathlib import Path


# Base configuration template (from filter_comparison_ema.json)
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
        "LEAVE_SUBJECT_OUT": 1,  # Will be overridden per subject
        "APPLY_FILTER": True,
        "FILTER_TYPE": "ema",
        "FILTER_ALPHA": 0.3,
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
        "WEIGHTS": "",  # Will be set per subject
        "PRUNING": {
            "ENABLED": True,
            "METHOD": "ln_structured",
            "AMOUNT": 0.3,  # Will be overridden per experiment
            "NORM": 2,  # L2-norm
            "DIM": 0,   # Prune output dimension (neurons)
            "TARGET_LAYERS": ["fc1"]
        }
    },
    "TRAINING": {
        "BATCH_SIZE": 64,
        "EPOCHS": 10,  # Not used (fine-tuning schedule determined by prune_and_finetune.py)
        "CP_INTERVAL": 5,
        "LEARNING_RATE": 1e-4,  # Initial LR (will be managed by prune_and_finetune.py)
        "WEIGHT_DECAY": 5e-5,
        "EPSILON": 1e-8,
        "MOMENTUM": 0.999,
        "SCHEDULER_STEPS": 10000000,
        "GRADIENT_CLIP": 1.0,
        "BATCH_NUM_WORKERS": 2,
        "WEIGHTED_SAMPLING": False
    },
    "SYSTEM": {
        "GPU": 0
    },
    "OUTPUT_DIR": "",  # Will be set per experiment
    "RANDOM_SEED": 42,  # Will be overridden per experiment
    "STORE_CSV": False
}


# Experiment parameters
SUBJECTS = [2, 3, 6]
PRUNING_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5]
SEEDS = [42, 123, 456]

# Subject ID mapping for file paths
SUBJECT_ID_MAP = {
    2: "02",
    3: "03",
    6: "06"
}


def get_baseline_model_path(subject):
    """Get the path to the baseline EMA model weights for a subject.

    Args:
        subject (int): Subject number (2, 3, or 6)

    Returns:
        str: Path to model_weights_10.pt
    """
    subject_str = SUBJECT_ID_MAP[subject]
    return f"outputs/cnn_filter_comparison/ema_a03_s{subject_str}_a03/model_weights_10.pt"


def generate_config(subject, pruning_pct, seed):
    """Generate a single pruning config.

    Args:
        subject (int): Subject number (2, 3, or 6)
        pruning_pct (float): Pruning percentage (0.1 to 0.5)
        seed (int): Random seed

    Returns:
        dict: Configuration dictionary
    """
    config = json.loads(json.dumps(BASE_CONFIG))  # Deep copy

    # Set subject-specific parameters
    config["DATA"]["LEAVE_SUBJECT_OUT"] = subject
    config["MODEL"]["WEIGHTS"] = get_baseline_model_path(subject)

    # Set pruning parameters
    config["MODEL"]["PRUNING"]["AMOUNT"] = pruning_pct

    # Set random seed
    config["RANDOM_SEED"] = seed

    # Set output directory
    subject_str = SUBJECT_ID_MAP[subject]
    pruning_pct_str = f"{int(pruning_pct * 100)}pct"
    config["OUTPUT_DIR"] = f"outputs/pruning/ema_s{subject_str}_prune{pruning_pct_str}_seed{seed}"

    return config


def main():
    # Create output directory for configs
    config_dir = Path("config/pruning")
    config_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("GENERATING PRUNING CONFIGURATION FILES")
    print("=" * 80)
    print(f"\nOutput directory: {config_dir}")
    print(f"Total configs to generate: {len(SUBJECTS) * len(PRUNING_LEVELS) * len(SEEDS)}")
    print()

    configs_generated = []

    # Generate all combinations
    for subject in SUBJECTS:
        for pruning_pct in PRUNING_LEVELS:
            for seed in SEEDS:
                # Generate config
                config = generate_config(subject, pruning_pct, seed)

                # Create filename
                subject_str = SUBJECT_ID_MAP[subject]
                pruning_pct_str = f"{int(pruning_pct * 100)}pct"
                filename = f"prune_ema_s{subject_str}_{pruning_pct_str}_seed{seed}.json"
                filepath = config_dir / filename

                # Write to file
                with open(filepath, 'w') as f:
                    json.dump(config, f, indent=2)

                configs_generated.append(filename)
                print(f"+ Generated: {filename}")

    # Summary
    print()
    print("=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)
    print(f"Total configs generated: {len(configs_generated)}")
    print(f"Saved to: {config_dir.resolve()}")

    # Breakdown summary
    print("\nBreakdown:")
    print(f"  Subjects: {SUBJECTS}")
    print(f"  Pruning levels: {[f'{int(p*100)}%' for p in PRUNING_LEVELS]}")
    print(f"  Random seeds: {SEEDS}")

    # Example configs
    print("\nExample configurations:")
    for i, filename in enumerate(configs_generated[:3]):
        print(f"  {i+1}. {filename}")
    print(f"  ... ({len(configs_generated) - 3} more)")

    # Verify baseline models exist
    print("\n" + "=" * 80)
    print("VERIFYING BASELINE MODEL PATHS")
    print("=" * 80)

    missing_models = []
    for subject in SUBJECTS:
        model_path = get_baseline_model_path(subject)
        if os.path.exists(model_path):
            print(f"+ Found: {model_path}")
        else:
            print(f"- MISSING: {model_path}")
            missing_models.append(model_path)

    if missing_models:
        print(f"\n! WARNING: {len(missing_models)} baseline model(s) not found!")
        print("Please ensure these models exist before running experiments.")
    else:
        print("\n+ All baseline models found!")

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("\n1. Test a single experiment:")
    print(f"   python scripts/06_model_compression/prune_and_finetune.py \\")
    print(f"       --json config/pruning/{configs_generated[0]}")
    print("\n2. Run full sweep on HPC:")
    print("   sbatch scripts/06_model_compression/run_pruning_sweep.sh")
    print()


if __name__ == '__main__':
    main()
