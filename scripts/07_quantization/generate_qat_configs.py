#!/usr/bin/env python3
"""
Generate QAT Configuration Files

Creates JSON configuration files for Quantization-Aware Training (QAT) on pruned models.
Generates configs for subjects 2, 3, and 6 using seed42 (best performer from pruning).

Usage:
    python generate_qat_configs.py
"""

import json
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
        "LEAVE_SUBJECT_OUT": None,  # Will be set per subject
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
        "WEIGHTS": None,  # Will be set per subject
        "QUANTIZATION": {
            "ENABLED": True,
            "METHOD": "qat",  # Quantization-Aware Training
            "BACKEND": "qnnpack",  # Optimized for mobile/embedded
            "DTYPE": "qint8"  # INT8 quantization
        }
    },
    "TRAINING": {
        "BATCH_SIZE": 64,
        "EPOCHS": 5,  # QAT fine-tuning epochs
        "CP_INTERVAL": 5,
        "LEARNING_RATE": 1e-5,  # Low LR to avoid catastrophic forgetting
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
    "OUTPUT_DIR": None,  # Will be set per subject
    "RANDOM_SEED": 42,
    "STORE_CSV": False
}

def generate_config(subject_id, seed=42):
    """
    Generate a QAT config for a specific subject.

    Args:
        subject_id: Subject ID (2, 3, or 6)
        seed: Random seed (default: 42)

    Returns:
        dict: Configuration dictionary
    """
    config = json.loads(json.dumps(BASE_CONFIG))  # Deep copy

    # Set subject-specific values
    config["DATA"]["LEAVE_SUBJECT_OUT"] = subject_id
    config["MODEL"]["WEIGHTS"] = f"outputs/pruning/ema_s0{subject_id}_prune40pct_seed{seed}/pruned_final.pt"
    config["OUTPUT_DIR"] = f"outputs/quantization/qat_s0{subject_id}_seed{seed}"
    config["RANDOM_SEED"] = seed

    return config

def main():
    """Generate QAT config files for subjects 2, 3, and 6 with seed42."""

    # Create config directory
    config_dir = Path("config/quantization")
    config_dir.mkdir(parents=True, exist_ok=True)

    subjects = [2, 3, 6]
    seed = 42

    print("=" * 70)
    print("Generating QAT Configuration Files")
    print("=" * 70)
    print(f"\nSubjects: {subjects}")
    print(f"Random Seed: {seed}")
    print(f"Output Directory: {config_dir}")
    print()

    for subject_id in subjects:
        # Generate config
        config = generate_config(subject_id, seed)

        # Save to file
        config_filename = f"qat_s0{subject_id}_seed{seed}.json"
        config_path = config_dir / config_filename

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"✓ Generated: {config_path}")
        print(f"  - Leave-out subject: {subject_id}")
        print(f"  - Pruned model: {config['MODEL']['WEIGHTS']}")
        print(f"  - Output: {config['OUTPUT_DIR']}")
        print()

    print("=" * 70)
    print(f"✓ All {len(subjects)} config files generated successfully!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Run: python scripts/07_quantization/quantize_qat.py --config config/quantization/qat_s02_seed42.json")
    print("  2. Run: python scripts/07_quantization/quantize_qat.py --config config/quantization/qat_s03_seed42.json")
    print("  3. Run: python scripts/07_quantization/quantize_qat.py --config config/quantization/qat_s06_seed42.json")
    print()

if __name__ == "__main__":
    main()
