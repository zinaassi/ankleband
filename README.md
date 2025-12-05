# ML PROJECT ADDITIONS

# Running Filter Experiments on HPC

This guide covers running systematic filter frequency experiments on the HPC cluster to evaluate different Butterworth low-pass filter cutoff frequencies for IMU signal preprocessing.

## Overview

The experiment framework tests multiple filter configurations to find the optimal cutoff frequency for gesture classification. It includes:

- **7 experiments**: Baseline (no filter) + 6 different cutoff frequencies (10, 12, 15, 18, 20, 25 Hz)
- **Automated job submission**: Sequential or parallel execution
- **Results collection**: Automated comparison and visualization
- **GPU acceleration**: Utilizes HPC GPU resources for training

---

## File Structure

```
ankleband/
├── config/bracelet/          # Configuration files
│   ├── hpc_baseline.json
│   ├── hpc_filter_10hz.json
│   ├── hpc_filter_12hz.json
│   └── ...
├── generate_configs.py       # Config file generator
├── run_sequential.sh         # Sequential job submission
├── run_single_job.sh         # Single experiment job script
├── run_array_job.sh          # Parallel job array (optional)
├── test_env.sh              # Environment testing script
├── collect_results.py        # Results collection and analysis
├── logs/                     # Job output logs
│   ├── exp_JOBID.out        # Standard output
│   └── exp_JOBID.err        # Error output
└── outputs/                  # Experiment results
    ├── hpc_baseline/
    ├── hpc_filter_10hz/
    └── ...
```

---

## Prerequisites

### 1. HPC Access

- SSH access to HPC cluster
- SLURM job scheduler available
- GPU partition access

### 2. Environment Setup

```bash
# Activate conda environment
conda activate imugr

# Install required packages
pip install fastdtw swifter pandas numpy scipy h5py torch matplotlib

# Verify installations
python -c "import fastdtw; import swifter; import torch; print('✓ All packages installed successfully')"
```

### 3. Test Environment (Recommended)

Before running full experiments, test that the environment works on the HPC:

```bash
# Submit test job
sbatch test_env.sh

# Check results after ~30 seconds
cat logs/test_env.out

# Should see:
# ✓ fastdtw imported successfully
# ✓ torch imported successfully
#   CUDA available: True
#   GPU: NVIDIA GeForce RTX 2080 Ti
```

---

## Configuration Files

### Automatic Generation

Use the provided script to generate all configuration files:

```bash
python generate_configs.py
```

This creates 7 configuration files in `config/bracelet/`:

| Config File | Description | Filter Settings |
|------------|-------------|-----------------|
| `hpc_baseline.json` | No filtering (baseline) | `APPLY_FILTER: false` |
| `hpc_filter_10hz.json` | 10 Hz cutoff | `FILTER_CUTOFF: 10`, `FILTER_ORDER: 5` |
| `hpc_filter_12hz.json` | 12 Hz cutoff | `FILTER_CUTOFF: 12`, `FILTER_ORDER: 5` |
| `hpc_filter_15hz.json` | 15 Hz cutoff | `FILTER_CUTOFF: 15`, `FILTER_ORDER: 5` |
| `hpc_filter_18hz.json` | 18 Hz cutoff | `FILTER_CUTOFF: 18`, `FILTER_ORDER: 5` |
| `hpc_filter_20hz.json` | 20 Hz cutoff | `FILTER_CUTOFF: 20`, `FILTER_ORDER: 5` |
| `hpc_filter_25hz.json` | 25 Hz cutoff | `FILTER_CUTOFF: 25`, `FILTER_ORDER: 5` |

### Manual Configuration

To create custom configurations, copy and modify an existing config file:

```bash
cp config/bracelet/hpc_baseline.json config/bracelet/custom_config.json
# Edit custom_config.json with your parameters
```

Key parameters to modify:
- `FILTER_CUTOFF`: Cutoff frequency in Hz
- `FILTER_ORDER`: Filter order (typically 5)
- `EPOCHS`: Number of training epochs (default: 10)
- `BATCH_SIZE`: Training batch size (default: 64)
- `OUTPUT_DIR`: Where to save results

---

## Running Experiments

### Method 1: Sequential Execution (Recommended)

Runs experiments one at a time automatically. This is the most reliable method and respects HPC job limits.

```bash
# Create logs directory
mkdir -p logs

# Make scripts executable
chmod +x run_sequential.sh run_single_job.sh

# Submit all experiments
./run_sequential.sh
```

**Output:**
```
Submitting sequential jobs...
Submitted hpc_baseline.json - Job ID: 329746
Submitted hpc_filter_10hz.json - Job ID: 329747 (depends on previous)
Submitted hpc_filter_12hz.json - Job ID: 329748 (depends on previous)
...
All jobs submitted in sequence!
```

**Features:**
- ✅ Only 1 job runs at a time
- ✅ Automatic chaining (next job starts when previous finishes)
- ✅ Respects HPC submission limits
- ✅ No manual intervention needed

### Method 2: Job Array (Parallel Execution)

Runs multiple experiments simultaneously if your HPC allows and has enough GPUs.

```bash
# Submit job array
sbatch run_array_job.sh

# This submits all 7 jobs at once
```

**Note:** This may fail with `QOSMaxSubmitJobPerUserLimit` if your HPC restricts concurrent job submissions.

### Method 3: Manual Submission

Submit individual experiments one by one:

```bash
# Submit specific experiments
sbatch run_single_job.sh hpc_baseline.json
sbatch run_single_job.sh hpc_filter_12hz.json
sbatch run_single_job.sh hpc_filter_15hz.json

# Check status
squeue -u $USER
```

---

## Monitoring Jobs

### Check Job Status

```bash
# View all your jobs
squeue -u $USER

# Output shows:
# JOBID  PARTITION  NAME        USER      ST  TIME   NODES
# 329746 all        filter_exp  zina.assi R   5:23   1
# 329747 all        filter_exp  zina.assi PD  0:00   1
```

**Job States:**
- `R` = Running
- `PD` = Pending (waiting to start)
- `CG` = Completing

### Watch Job Progress

```bash
# Auto-refresh job queue every 10 seconds
watch -n 10 squeue -u $USER

# Monitor training logs in real-time
tail -f logs/exp_329746.out

# Check for errors
tail -f logs/exp_329746.err
```

### View Completed Jobs

```bash
# Jobs completed today
sacct -u $USER --starttime=today

# Detailed job information
sacct -u $USER --format=JobID,JobName,State,Elapsed,ExitCode --starttime=today
```

### Expected Training Output

A successful training run should show:

```
Config: hpc_baseline.json
Starting at: Fri 05 Dec 2025 11:00:00 PM IST
Python: /home/zina.assi/miniconda3/envs/imugr/bin/python
Conda env: imugr

Loading dataset...
Found 21 H5 files
Processing ID01_seating_all_gestures.h5...
...

Creating model...
Model parameters: 14,234

Starting training...
Epoch 1/10: Train Loss=0.523, Train Acc=0.845, Val Acc=0.812
Epoch 2/10: Train Loss=0.412, Train Acc=0.892, Val Acc=0.867
...
Epoch 10/10: Train Loss=0.145, Train Acc=0.972, Val Acc=0.948

Training complete!
Final test accuracy: 0.953
Finished at: Fri 05 Dec 2025 11:45:32 PM IST
```

---

## Disconnecting During Experiments

**Good news:** You can safely disconnect after submitting jobs!

```bash
# Submit jobs
./run_sequential.sh

# You can now:
# - Close your laptop ✓
# - Disconnect from WiFi ✓
# - Turn off your computer ✓

# Jobs continue running on HPC servers
```

**When you reconnect:**

```bash
# SSH back to HPC
ssh your_username@hpc_address

# Check job status
squeue -u $USER

# View logs
tail logs/exp_*.out

# Check results
ls -lh outputs/
```

---

## Collecting Results

### Automated Collection

After all experiments complete:

```bash
# Run results collector
python collect_results.py
```

This generates:
- `frequency_comparison_results.csv` - Tabular results
- `frequency_comparison.png` - Visualization plots

**Example output:**
```
Collecting results from experiments...
============================================================
✓ Baseline (No Filter): Acc=0.948, Rec=0.945, Prec=0.951
✓ 10 Hz: Acc=0.951, Rec=0.949, Prec=0.953
✓ 12 Hz: Acc=0.953, Rec=0.951, Prec=0.956
✓ 15 Hz: Acc=0.950, Rec=0.948, Prec=0.952
✓ 18 Hz: Acc=0.949, Rec=0.946, Prec=0.951
✓ 20 Hz: Acc=0.948, Rec=0.945, Prec=0.950
✓ 25 Hz: Acc=0.947, Rec=0.944, Prec=0.949
============================================================

BEST RESULTS:
============================================================
Best Accuracy:  12 Hz (0.953)
Best Recall:    12 Hz (0.951)
Best Precision: 12 Hz (0.956)
```

### Manual Results Review

```bash
# List all output directories
ls -lh outputs/

# View results for specific experiment
cat outputs/hpc_filter_12hz/results.csv

# Check all result files
find outputs/ -name "*.csv"
```

### Copy Results to Local Machine

From your local computer:

```bash
# Copy entire results directory
scp -r username@hpc:/path/to/ankleband/outputs ./hpc_results/

# Copy comparison files
scp username@hpc:/path/to/ankleband/frequency_comparison* ./

# Copy specific experiment
scp -r username@hpc:/path/to/ankleband/outputs/hpc_filter_12hz ./
```

---

## Job Script Details

### `run_single_job.sh`

Main job script for running a single experiment:

```bash
#!/bin/bash
#SBATCH --job-name=filter_exp       # Job name
#SBATCH --output=logs/exp_%j.out    # Standard output (%j = job ID)
#SBATCH --error=logs/exp_%j.err     # Error output
#SBATCH --time=8:00:00              # Max runtime (8 hours)
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --mem=32G                   # Memory allocation
#SBATCH --cpus-per-task=2           # CPU cores

CONFIG_FILE=$1  # Config file passed as argument

module load anaconda3
module load cuda/12.4

eval "$(conda shell.bash hook)"
conda activate imugr

cd $HOME/ankleband
python trainer/train_conv.py --json config/bracelet/$CONFIG_FILE
```

**Parameters to adjust:**
- `--time`: Increase if training takes longer
- `--mem`: Increase if you get out-of-memory errors
- `--cpus-per-task`: Reduce if you hit CPU limits

### `run_sequential.sh`

Master script that submits jobs in sequence:

```bash
#!/bin/bash

configs=(
    "hpc_baseline.json"
    "hpc_filter_10hz.json"
    # ... etc
)

# Submit first job
JOB_ID=$(sbatch --parsable run_single_job.sh "${configs[0]}")

# Submit remaining jobs with dependencies
for i in {1..6}; do
    JOB_ID=$(sbatch --parsable --dependency=afterok:$JOB_ID \
             run_single_job.sh "${configs[$i]}")
done
```

---

## Troubleshooting

### Job Fails Immediately (Runs <10 seconds)

**Symptoms:** Job finishes in 5-10 seconds, no training output

**Check error log:**
```bash
cat logs/exp_JOBID.err
```

**Common causes:**

1. **Missing Python packages**
   ```
   ModuleNotFoundError: No module named 'fastdtw'
   ```
   **Fix:**
   ```bash
   conda activate imugr
   pip install fastdtw swifter
   ```

2. **Wrong config file path**
   ```
   FileNotFoundError: config/bracelet/hpc_baseline.json
   ```
   **Fix:**
   ```bash
   python generate_configs.py  # Regenerate configs
   ```

3. **Dataset not found**
   ```
   FileNotFoundError: data/dataset/ID01_seating_all_gestures.h5
   ```
   **Fix:** Verify dataset is in correct location

### QOSMaxSubmitJobPerUserLimit Error

**Symptoms:**
```
sbatch: error: QOSMaxSubmitJobPerUserLimit
sbatch: error: Batch job submission failed
```

**Cause:** Too many jobs submitted at once

**Solution 1:** Use sequential submission (recommended)
```bash
./run_sequential.sh
```

**Solution 2:** Submit in smaller batches
```bash
# Submit first 3
sbatch run_single_job.sh hpc_baseline.json
sbatch run_single_job.sh hpc_filter_10hz.json
sbatch run_single_job.sh hpc_filter_12hz.json

# Wait for these to finish, then submit more
```

**Solution 3:** Wait for current jobs to complete
```bash
squeue -u $USER  # Check running jobs
# Wait for some to finish before submitting more
```

### QOSMaxCpuPerUserLimit Error

**Symptoms:** Job stays pending with reason `QOSMaxCpuPerUserLimit`

**Cause:** Requesting too many CPUs

**Fix:** Reduce CPUs in `run_single_job.sh`:
```bash
#SBATCH --cpus-per-task=2  # Change from 4 to 2
# or even
#SBATCH --cpus-per-task=1  # Try 1 if still issues
```

### Job Pending Forever

**Check why it's pending:**
```bash
squeue -u $USER
# Look at REASON column
```

**Common reasons:**

- `(Resources)` - Waiting for available GPU/CPU
- `(Priority)` - Other users have higher priority
- `(Dependency)` - Waiting for previous job (expected in sequential mode)
- `(QOSMaxCpuPerUserLimit)` - Too many CPUs requested (see above)

**Check available resources:**
```bash
sinfo -o "%20P %5a %.10l %16F %G"
```

### Training Crashes Mid-Run

**Check error log for:**

1. **Out of memory**
   ```
   CUDA out of memory
   ```
   **Fix:** Reduce batch size in config file:
   ```json
   "TRAINING": {
       "BATCH_SIZE": 32  // Reduce from 64
   }
   ```

2. **GPU error**
   ```
   RuntimeError: CUDA error
   ```
   **Fix:** Check GPU availability, try different node

### Environment Activation Issues

**Symptoms:**
```
/var/spool/slurmd/jobXXXXX/slurm_script: line 17: activate: No such file or directory
```

**Fix:** Use proper conda activation in `run_single_job.sh`:
```bash
# Add this line before conda activate:
eval "$(conda shell.bash hook)"
```

---

## Performance Benchmarks

### Expected Runtime

Based on typical datasets (21 H5 files, 10 subjects):

| Configuration | Time per Epoch | Total Time (10 epochs) |
|--------------|----------------|------------------------|
| Baseline | ~3 minutes | ~30 minutes |
| With filtering | ~3.5 minutes | ~35 minutes |

**Factors affecting runtime:**
- Dataset size
- GPU type (2080 Ti vs Titan XP)
- Batch size
- Number of subjects in training

### Resource Usage

**Typical job:**
- GPU memory: ~4-6 GB
- System RAM: ~16-24 GB
- CPU usage: 1-2 cores
- Storage: ~500 MB per experiment output

---

## Advanced Usage

### Custom Filter Configurations

Test additional cutoff frequencies:

```bash
# Create custom config
cp config/bracelet/hpc_filter_12hz.json config/bracelet/hpc_filter_14hz.json

# Edit the new file
nano config/bracelet/hpc_filter_14hz.json
# Change: "FILTER_CUTOFF": 14
# Change: "OUTPUT_DIR": "outputs/hpc_filter_14hz"

# Submit
sbatch run_single_job.sh hpc_filter_14hz.json
```

### Testing Different Filter Orders

Modify config to test filter order impact:

```json
"DATA": {
    "FILTER_ORDER": 3,  // Try 3, 5, 7, 9
    "FILTER_CUTOFF": 12
}
```

### Quick Testing with Subset

For rapid testing, use fewer subjects/epochs:

```json
"DATA": {
    "TRAIN_FILES": [
        "ID01_seating_all_gestures.h5",
        "ID01_standing_all_gestures.h5"
    ],  // Just 1 subject
    "STRIDE": 5  // Skip more data
},
"TRAINING": {
    "EPOCHS": 3  // Quick test
}
```

### Debugging Mode

Add verbose output to job script:

```bash
# In run_single_job.sh, add:
set -x  # Print each command
python trainer/train_conv.py --json config/bracelet/$CONFIG_FILE --verbose
```

---

## Best Practices

### Before Running Full Experiments

1. ✅ Test environment: `sbatch test_env.sh`
2. ✅ Generate configs: `python generate_configs.py`
3. ✅ Run one quick test: `sbatch run_single_job.sh hpc_baseline.json`
4. ✅ Verify output structure: `ls -lh outputs/hpc_baseline/`
5. ✅ Then run full suite: `./run_sequential.sh`

### During Experiments

1. ✅ Monitor first job to ensure it runs correctly
2. ✅ Can disconnect after verifying first job starts
3. ✅ Check back periodically: `squeue -u $USER`
4. ✅ Review logs if any job fails

### After Experiments

1. ✅ Collect results: `python collect_results.py`
2. ✅ Backup outputs: Copy to local machine
3. ✅ Clean up logs: Archive or delete old log files
4. ✅ Document findings in your report

---

## Quick Reference

### Essential Commands

```bash
# Submit experiments
./run_sequential.sh

# Check status
squeue -u $USER

# View logs
tail -f logs/exp_*.out

# Collect results
python collect_results.py

# Cancel all jobs
scancel -u $USER

# Cancel specific job
scancel JOBID
```

### File Locations

```bash
# Configs:     config/bracelet/hpc_*.json
# Job scripts: run_single_job.sh, run_sequential.sh
# Logs:        logs/exp_*.out, logs/exp_*.err
# Results:     outputs/hpc_*/
# Analysis:    frequency_comparison_results.csv
```

---

## Support and Contact

For issues specific to:
- **HPC cluster:** Contact your HPC support team
- **SLURM errors:** Check SLURM documentation or HPC admin
- **Code/training issues:** Review project documentation or contact supervisors

---

## Changelog

### v1.0 (December 2025)
- Initial HPC experiment framework
- Support for 7 filter configurations
- Sequential and array job submission
- Automated results collection


# Smart Ankleband for Plug-and-Play Hand-Prosthetic Control

Official code repository for [https://arxiv.org/abs/2503.17846](https://arxiv.org/abs/2503.17846).  
See our demo in action by watching the video [https://youtu.be/IUvm3WCvYG8](https://youtu.be/IUvm3WCvYG8).  

<img src="materials/demo.gif" alt="robotic_hand_demo" width="900"/>  

*A demo showing our smart ankleband with the robotic hand to perform a daily activity, such as pouring a liquid from a soda can into a paper cup.*

### Dataset  

Our dataset is publicly available! Please refer to this page for explanations and instructions on how to download:  
[Dataset explanation](DATASET.md)

### System Requirements  

* Operating system - The project was built using Ubuntu 22.04, but should work on Windows as well.
* GPU - Any Nvidia RTX GPU is sufficient (the model is tiny and require approximately 500MB of GPU memory).
* RAM - 32GB or higher is required for training purposes.

The robotic hand we used for this project is based on an opensource project provided by Haifa3D, a non-profit organization providing self-made 3d-printed solutions for those who need them. To rebuild the hand, the STL files and code to program the hand are in the following repositories:  
[https://github.com/Haifa3D/hand-mechanical-design](https://github.com/Haifa3D/hand-mechanical-design)  
[https://github.com/Haifa3D/hand-electronic-design](https://github.com/Haifa3D/hand-electronic-design)  

### Installation  
  
The project is based on Python 3.9 and PyTorch 2.6 with CUDA 12.4. All the necessary packages are in ```requirements.txt```. We recommend creating a virtual environment using Anaconda as follows:  
  
1) Download and install Anaconda Python from here:  
[https://www.anaconda.com/products/individual/](https://www.anaconda.com/products/individual/)  
  
2) Enter the following commands to create a virtual environment:  
```
conda create -n imugr python=3.9
conda activate imugr
pip install -r requirements.txt
```

For more information on how to manage conda environments, please refer to:  
[https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)  

### Overview  
  
Training and testing-related files are found in the trainer folder, where they are operated using configuration (JSON) files that are organized in the config directory. The data folder contains a data loader that is executed for both training and testing (```load_data.py```), and the evaluation folder contains all the necessary code to replicate the experiments presented in the paper (given outputs from the testing procedures).  

The model presented in the paper is found in ```trainer/models/conv1d_model.py``` under the ```Conv1DNet``` class, and the baselines are implemented in the ```train_classic.py``` that is in the same folder.  

**Training**  

All training sessions are executed using the same training file (```trainer/train_conv.py```), by calling different JSON files. An example of training the model on our dataset is as follows (in some situations we referred to the ankleband as a bracelet):

```  
python trainer/train_conv.py --json config/bracelet/regular_bracelet_leaveone.json  
```  

In case needed, you can overwrite some of the properties listed in the JSON file. For example, if we want to exclude subject 7 and leave it for testing (instead of subject no. 1), we can execute the script as follows:

```  
python trainer/train_conv.py --json config/bracelet/regular_bracelet_leaveone.json  --loo 7
``` 

However, these JSON files contain all the necessary properties and there is no need to feed the python script with additional arguments.  

**Testing**  

While some of the evaluation is done during training and the training scripts are programmed to write test metrics into the output folder, additional script files are provided to further evaluate the methods. For example, ```trainer/compute_cm.py``` can be used to compute confusion matrix, and ```trainer/compute_metrics.py``` is provided to compute the metrics presented in the paper on an specific set. Here are a few execution examples:

```  
python trainer/compute_cm.py --json config/bracelet/test_regular_bracelet.json  
python trainer/compute_metrics.py --json config/bracelet/test_regular_bracelet.json  
```  

### Real-time Implementation  

To convert the model for execution on an ESP32 board, we wrote a script that loads our model and write all trained parameters into a single C++ file so we can recreate the inference procedure on the ESP32 board. The file ```model_conversions/extract_model_weights.py``` loads the pytorch model and writes the result as ```model_weights.h```.

To install the code on the ESP32, we uploaded the code project (along with our trained model) to the ```rt_code``` folder. The main file is ```rt_code/execute_imu_gestures.ino```, the class that manages the neural network inference is in the files ```neural_network_engine.cpp``` and ```neural_network_engine.h```. The model weights are in ```model_weights_feb27_conv_c10_im64_fc2.h``` (which is basically the result of the script file ```model_conversions/extract_model_weights.py```).

Here are the required libraries to install in the Arduino IDE in order to compile the code for ESP32:

* Adafruit BNO08x (Version 1.25).
* Eigen (Version 0.3.2).
* ArduinoBLE - should be included in the ESP32 library by Espressif. 

### Acknowledgments  

This research was supported in part by the Technion Autonomous Systems Program (TASP), the Wynn Family Foundation, the David Himelberg Foundation, the Israeli Ministry of Science \& Technology grants No. 3-17385 and in part by the United States-Israel Binational Science Foundation (BSF) grants no. 2019703 and 2021643.

We also thank the subjects that participated in the creation of the dataset, and [Haifa3D](https://www.facebook.com/Haifa3d/), a non-profit organization providing self-made 3d-printed solutions, for their consulting and support through the research.  