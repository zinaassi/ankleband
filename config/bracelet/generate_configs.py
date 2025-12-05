"""
Generate configuration files for different cutoff frequencies
"""
import json
import os

# Base configuration
base_config = {
    "MODE": "regular",
    "DATA": {
        "PATH": "data/dataset",
        "TRAIN_FILES": [
            "ID01_seating_all_gestures.h5","ID01_standing_all_gestures.h5",
            "ID02_seating_all_gestures.h5","ID02_standing_all_gestures.h5",
            "ID03_seating_all_gestures.h5","ID03_standing_all_gestures.h5",
            "ID04_seating_all_gestures.h5","ID04_standing_all_gestures.h5",
            "ID05_seating_all_gestures.h5","ID05_standing_all_gestures.h5",
            "ID06_seating_all_gestures.h5","ID06_standing_all_gestures.h5",
            "ID07_seating_all_gestures.h5","ID07_standing_gestures12.h5","ID07_standing_gestures34free.h5",
            "ID08_seating_all_gestures.h5","ID08_standing_all_gestures.h5",
            "ID09_seating_all_gestures.h5","ID09_standing_all_gestures.h5",
            "ID10_seating_all_gestures.h5","ID10_standing_all_gestures.h5"
        ],
        "TEST_FILES": [],
        "APPEND": 60,
        "STEP": 2,
        "STRIDE": 2,
        "CLASSES": 5,
        "LABEL_PERCENTAGE": 0.5,
        "LEAVE_SUBJECT_OUT": 1,
        "SHUFFLE": False
    },
    "MODEL": {
        "TYPE": "neuralnet",
        "NUM_FC_LAYERS": 2
    },
    "TRAINING": {
        "BATCH_SIZE": 64,
        "EPOCHS": 10,
        "CP_INTERVAL": 5,
        "LEARNING_RATE": 1e-4,
        "WEIGHT_DECAY": 5e-5,
        "EPSILON": 1e-8,
        "MOMENTUM": 0.999,
        "SCHEDULER_STEPS": 10000000,
        "GRADIENT_CLIP": 1.0,
        "BATCH_NUM_WORKERS": 4,
        "WEIGHTED_SAMPLING": False
    },
    "SYSTEM": {
        "GPU": 0
    },
    "RANDOM_SEED": 2,
    "STORE_CSV": True
}

# Cutoff frequencies to test
cutoff_frequencies = [10, 12, 15, 18, 20, 25]

# Output directory
config_dir = "config/bracelet"
os.makedirs(config_dir, exist_ok=True)

print("Generating configuration files...")
print("=" * 60)

# Generate baseline (no filter)
config = base_config.copy()
config["DATA"] = base_config["DATA"].copy()
config["DATA"]["APPLY_FILTER"] = False
config["OUTPUT_DIR"] = "outputs/hpc_baseline"

filename = os.path.join(config_dir, "hpc_baseline.json")
with open(filename, 'w') as f:
    json.dump(config, f, indent=4)
print(f"✓ Created: {filename}")

# Generate configs for each cutoff frequency
for cutoff in cutoff_frequencies:
    config = base_config.copy()
    config["DATA"] = base_config["DATA"].copy()
    config["DATA"]["APPLY_FILTER"] = True
    config["DATA"]["FILTER_CUTOFF"] = cutoff
    config["DATA"]["FILTER_ORDER"] = 5
    config["OUTPUT_DIR"] = f"outputs/hpc_filter_{cutoff}hz"
    
    filename = os.path.join(config_dir, f"hpc_filter_{cutoff}hz.json")
    with open(filename, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"✓ Created: {filename}")

print("=" * 60)
print(f"Generated {len(cutoff_frequencies) + 1} configuration files!")
print("\nNext steps:")
print("1. Run: ./run_all_experiments.sh")
print("2. Monitor: squeue -u $USER")
print("3. Check logs: tail -f logs/exp_*.out")