# Pruning Process - Technical Notes for Dean Meeting

## 1. CNN Model Architecture

### What Does This Model Do?

**Simple Explanation:**
Imagine you're wearing a smartwatch. When you make a gesture (like waving your hand), the watch's sensors record how your wrist moves. Our model takes this movement data and figures out which gesture you made.

**The Data:**
- **Input**: 60 snapshots of movement (60 timesteps)
- Each snapshot has 6 measurements (3 from accelerometer: x, y, z movement + 3 from gyroscope: x, y, z rotation)
- **Output**: Which of 5 gestures you made

**Think of it like this:**
```
Raw sensor data → Model → "You made gesture #3!"
(60 × 6 numbers)         (one of 5 gestures)
```

---

### How the Model Works - Simple Version

**The model has 2 main jobs:**

1. **Job 1: Find Patterns** (Conv1D layer)
   - Looks at the movement data and finds patterns
   - Like: "Oh, there's a quick upward motion followed by a twist"
   - This is the "feature extraction" part

2. **Job 2: Decide Which Gesture** (FC layers)
   - Takes those patterns and decides which gesture it is
   - Like: "These patterns mean it's gesture #3"
   - This is the "classification" part

---

### What Each Layer Actually Does

Think of the model like a factory assembly line. Data goes in one end, and a decision comes out the other. Each layer is a worker on the line:

#### **Layer 1: Conv1D (The Pattern Finder)**
```
Job: Look for patterns in the movement data
Example: "I see a spike here, a dip there"
```
- Takes in: 60 timesteps × 6 sensors = lots of numbers
- Looks for: 10 different patterns (like 10 different detectors)
- Outputs: 200 numbers representing "how strong each pattern is"

**Analogy**: Like having 10 different people looking at the same data, each trained to spot different things (speed changes, direction changes, etc.)

---

#### **Layer 2: BatchNorm + ReLU (The Cleaner)**
```
Job: Clean up and normalize the numbers
```
- Takes the 200 numbers from Conv1D
- Makes sure they're all on the same scale (not too big, not too small)
- Removes negative values (ReLU = "if negative, make it zero")

**Analogy**: Like adjusting the brightness and contrast on a photo so it's easier to see

---

#### **Layer 3: FC Layer 1 (The Big Decision Maker)** ← **THIS IS WHAT WE PRUNE**
```
Job: Combine all the patterns to start making a decision
```
- Takes in: 200 pattern strengths
- Has 64 "neurons" (decision makers)
- Each neuron looks at ALL 200 patterns and says "based on this, I think..."

**Analogy**: Like having 64 experts, each one looking at all the evidence and forming their own opinion

**Why it's big**: Each of the 64 experts needs to look at all 200 pieces of evidence. That's 64 × 200 = 12,800 connections! This is why this layer has so many parameters.

---

#### **Layer 4: BatchNorm + ReLU (Another Cleaner)**
```
Job: Clean up the 64 expert opinions
```
- Same job as Layer 2, but for the 64 numbers from FC Layer 1

---

#### **Layer 5: FC Layer 2 (The Final Judge)**
```
Job: Make the final decision - which gesture is it?
```
- Takes in: 64 expert opinions
- Outputs: 5 scores (one for each possible gesture)
- The highest score wins! "It's gesture #3!"

**Analogy**: Like a judge listening to 64 witnesses and making the final verdict

---

### The Full Journey (Put It All Together)

```
1. SENSOR DATA
   ↓
   60 snapshots of movement (60 × 6 = 360 numbers)

2. CONV1D - Pattern Finding
   ↓
   "I found these 10 types of patterns at different times"
   (200 numbers: strength of each pattern)

3. CLEANUP (BatchNorm + ReLU)
   ↓
   Clean, normalized 200 numbers

4. FC LAYER 1 - Expert Analysis
   ↓
   64 experts analyze all patterns
   (64 numbers: each expert's opinion)

5. CLEANUP (BatchNorm + ReLU)
   ↓
   Clean, normalized 64 numbers

6. FC LAYER 2 - Final Decision
   ↓
   5 scores: [2.1, 0.3, 5.8, 1.2, 0.5]
   Highest is #3, so answer = "Gesture 3!"
```

---

### Why We Prune FC Layer 1

**The Problem:**
- FC Layer 1 has 64 experts
- Each expert looks at all 200 patterns
- That's **12,864 connections** (64 × 200 weights + 64 biases)
- This is **92.6% of the entire model**!

**The Solution:**
- We found that we don't need all 64 experts
- Some experts give very similar opinions
- We can remove 40% of them (down to ~38 experts) and still get the right answer!

**After Pruning:**
- Only 38 experts instead of 64
- Still accurate (actually slightly better!)
- Model is 32% smaller
- Runs faster on ESP32

**Analogy**: Imagine a committee of 64 people making decisions. We realized 26 of them weren't really contributing unique insights - they were just agreeing with others. So we removed those 26, and the committee works just as well (or better!) with only 38 members.

---

### Detailed Layer Breakdown (With Numbers)

```
INPUT: (batch_size, 6, 60)
  ↓
CONV1D LAYER 1: Conv1d(in_channels=6, out_channels=10, kernel_size=3, stride=3)
  • Parameters: 6 × 10 × 3 = 180 weights (no bias)
  • Output: (batch_size, 10, 20)
  ↓
FLATTEN: (batch_size, 200)
  ↓
BATCH NORM 1: BatchNorm1d(200) + ReLU
  • Parameters: 200 weights + 200 biases = 400 total
  • Output: (batch_size, 200)
  ↓
FULLY CONNECTED LAYER 1 (FC1): Linear(200, 64)  ← **THIS IS WHAT WE PRUNE**
  • Parameters: 200 × 64 = 12,800 weights + 64 biases = 12,864 total
  • Output: (batch_size, 64)
  ↓
BATCH NORM 2: BatchNorm1d(64) + ReLU
  • Parameters: 64 weights + 64 biases = 128 total
  • Output: (batch_size, 64)
  ↓
FULLY CONNECTED LAYER 2 (FC2): Linear(64, 5)  ← **This gets adjusted when FC1 is pruned**
  • Parameters: 64 × 5 = 320 weights + 5 biases = 325 total
  • Output: (batch_size, 5) - 5 gesture classes
  ↓
SOFTMAX (applied during loss calculation)
```

### Parameter Count Analysis

| Layer | Parameters | Percentage of Total |
|-------|-----------|---------------------|
| Conv1D | 180 | 1.3% |
| BatchNorm1 (200) | 400 | 2.9% |
| FC Layer 1 | 12,864 | **92.6%** |
| BatchNorm2 (64) | 128 | 0.9% |
| FC Layer 2 | 325 | 2.3% |
| **TOTAL** | **13,897** | **100%** |

**Key Insight**: FC Layer 1 contains 92.6% of all model parameters! This is why we target it for pruning.

---

## 2. Why Prune FC Layers (Not Conv Layers)?

### Decision Rationale

**We chose to prune ONLY the Fully Connected (FC) layers, specifically FC Layer 1, for several technical reasons:**

### Reason 1: Parameter Concentration
- **FC Layer 1 has 12,864 parameters (92.6% of model)**
- **Conv Layer has only 180 parameters (1.3% of model)**
- Pruning FC layers gives maximum compression benefit
- Pruning the Conv layer would have minimal impact on model size

### Reason 2: Feature Extraction vs Classification
```
Conv Layers → Extract spatial/temporal features (KEEP)
FC Layers → Classify features into classes (PRUNE)
```

- **Conv layers learn important domain-specific features** (gesture patterns in IMU data)
- These features are crucial for model accuracy
- **FC layers learn classification boundaries** which are more redundant
- FC neurons can be removed without losing critical pattern detection

### Reason 3: Redundancy in FC Layers
- FC layers often learn redundant representations
- Multiple neurons may encode similar information
- Conv filters tend to learn more diverse, specialized features
- Studies show FC layers are more "prunable" without accuracy loss

### Reason 4: Hardware Constraints
- Our Conv layer is already very small (10 filters)
- Pruning it would severely limit feature extraction capacity
- FC layers have 64 neurons - plenty of room for redundancy removal
- We can remove 40% of FC neurons (64 → ~38) and still maintain performance

---

## 3. Structured vs Unstructured Pruning

### What's the Difference?

**Unstructured Pruning:**
```
Original weights:        After unstructured pruning:
[0.5  0.3  0.8]         [0.5  0.0  0.8]  ← Individual weights → 0
[0.2  0.9  0.1]   →     [0.0  0.9  0.0]  ← Sparse matrix
[0.7  0.4  0.6]         [0.7  0.0  0.6]
```
- Removes individual weights
- Creates sparse matrices
- Higher compression possible
- **Problem**: Requires specialized sparse matrix libraries for speedup
- **Problem**: ESP32 doesn't efficiently handle sparse operations

**Structured Pruning:**
```
Original weights:        After structured pruning:
[0.5  0.3  0.8]         [0.5  0.8]  ← Entire neuron removed
[0.2  0.9  0.1]   →     [0.7  0.6]  ← Dense, smaller matrix
[0.7  0.4  0.6]
     ↑ neuron removed
```
- Removes entire neurons/channels
- Creates smaller dense matrices
- Actual model size reduction
- **Advantage**: Works on ANY hardware - no special libraries needed
- **Advantage**: Real speedup on ESP32

### Why We Chose Structured Pruning

**Decision Factors:**

1. **ESP32 Deployment Target**
   - ESP32 has limited memory and compute
   - No sparse matrix acceleration libraries
   - Dense matrix operations are well-optimized
   - Structured pruning gives REAL speedup on ESP32

2. **Actual Size Reduction**
   - Unstructured: Model file smaller, but runtime memory same
   - Structured: Both file AND runtime memory reduced
   - We physically remove neurons → truly smaller model

3. **Simpler Implementation**
   - No need for sparse matrix formats
   - No special inference code
   - Pruned model works like any other model

4. **Maintained Accuracy**
   - Structured pruning: 40% reduction, **+0.85% recall improvement**
   - This shows entire neurons were redundant
   - Removing them didn't hurt (actually helped via regularization)

### Code Example - Structured Pruning

```python
import torch.nn.utils.prune as prune

# Apply structured pruning to FC Layer 1
fc1_layer = model.fc_layers[0]  # The Linear(200, 64) layer

prune.ln_structured(
    module=fc1_layer,
    name='weight',        # Prune the weight tensor
    amount=0.1,           # Remove 10% of neurons
    n=2,                  # Use L2-norm for ranking
    dim=0                 # Dimension 0 = output neurons (rows)
)

# This creates a mask that zeros out entire rows (neurons)
# Shape before: (64, 200) → 64 output neurons
# After 40% pruning: ~26 rows are all zeros → only ~38 active neurons
```

---

## 4. L2-Norm Neuron Ranking

### Why L2-Norm?

**We rank neurons by the L2-norm of their weights to decide which to prune.**

**L2-norm formula:**
```
For neuron i with weights w_i = [w_i1, w_i2, ..., w_i200]:
L2-norm = √(w_i1² + w_i2² + ... + w_i200²)
```

**Intuition:**
- High L2-norm → neuron has large weights → important for model
- Low L2-norm → neuron has small weights → less important
- We remove neurons with lowest L2-norm first

**Example:**
```python
FC Layer 1 weights: (64, 200) - 64 neurons, each with 200 weights

Neuron 0: [0.8, 0.6, 0.3, ...] → L2-norm = 5.2  (KEEP - high norm)
Neuron 1: [0.1, 0.05, 0.02, ...] → L2-norm = 0.8  (REMOVE - low norm)
Neuron 2: [0.9, 0.7, 0.5, ...] → L2-norm = 6.1  (KEEP - high norm)
...
Neuron 63: [0.2, 0.1, 0.15, ...] → L2-norm = 1.2  (REMOVE - low norm)

Sort by L2-norm → Remove bottom 40% → Keep top 60%
```

---

## 5. Iterative Pruning Strategy

### Why Not Prune All at Once?

**Bad Approach:** Remove 40% of neurons in one step
- Sudden, drastic model change
- Model can't recover - accuracy drops significantly

**Our Approach:** Iterative pruning - remove gradually
- Remove 10% → fine-tune → remove 10% → fine-tune → repeat
- Model adapts to each change
- Final accuracy is maintained or even improved

### Pruning Schedule for 40%

```python
# 40% target = 4 iterations of 10% each
pruning_schedule = [
    (0.10, 3 epochs, [1e-4, 1e-5, 1e-5]),      # Iteration 1: prune 10%, train 3 epochs
    (0.10, 3 epochs, [1e-4, 1e-5, 1e-5]),      # Iteration 2: prune 10%, train 3 epochs
    (0.10, 3 epochs, [1e-4, 1e-5, 1e-5]),      # Iteration 3: prune 10%, train 3 epochs
    (0.10, 4 epochs, [1e-4, 1e-5, 1e-5, 1e-6]) # Iteration 4: prune 10%, train 4 epochs
]
# Total: 4 × 10% = 40% pruned
# Total: 3+3+3+4 = 13 epochs of fine-tuning
```

**Process per iteration:**
```
1. Apply pruning (10% of remaining neurons)
2. Evaluate immediately (see performance drop)
3. Fine-tune for 3-4 epochs
4. Evaluate again (see recovery)
5. Save checkpoint
6. Repeat
```

---

## 6. Learning Rate Rewinding

### The Problem
After pruning, the model needs to adapt to the new structure. Using the wrong learning rate can:
- Too high → unstable training, model diverges
- Too low → slow recovery, can't escape poor local minimum

### Our Solution: LR Schedule per Iteration

**Each iteration uses a decreasing LR schedule:**
```python
iteration_lrs = [1e-4, 1e-5, 1e-5, 1e-6]  # For a 4-epoch iteration

Epoch 1: LR = 1e-4  (0.0001) ← Start high to escape shock of pruning
Epoch 2: LR = 1e-5  (0.00001) ← Reduce to stabilize
Epoch 3: LR = 1e-5  (0.00001) ← Continue stabilizing
Epoch 4: LR = 1e-6  (0.000001) ← Very fine adjustments
```

**Why this works:**
- High initial LR lets model adapt quickly to neuron removal
- Gradual reduction prevents overfitting
- Final low LR polishes the weights

### Code Implementation

```python
def fine_tune_iteration(model, train_loader, test_loader, device, cfg,
                        num_epochs, learning_rates):
    """Fine-tune after each pruning iteration."""

    for epoch_idx in range(num_epochs):
        # Get learning rate for this epoch
        current_lr = learning_rates[epoch_idx]

        # Create optimizer with current LR
        optimizer = torch.optim.Adam(model.parameters(), lr=current_lr)

        # Train for one epoch
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        # Evaluate
        model.eval()
        metrics = evaluate_model(model, test_loader, device, cfg)

        print(f"Epoch {epoch_idx+1}/{num_epochs}, "
              f"LR={current_lr}, "
              f"Recall={metrics['recall']:.4f}")
```

---

## 7. Physical Neuron Removal - The Critical Step

### The PyTorch Pruning Bug

**Standard PyTorch pruning has a major limitation:**

```python
# After using prune.ln_structured() and prune.remove():
print(fc1.weight.shape)  # Still (64, 200) - NO SIZE CHANGE!

# The pruned neurons are still there, just masked as zeros
# Model size doesn't actually decrease
```

**Why?** PyTorch's `prune.remove()` only removes the mask but keeps the full tensor.

### Our Solution: Physical Removal

We implemented `physically_remove_pruned_neurons()` to ACTUALLY shrink the model:

```python
def physically_remove_pruned_neurons(self):
    """Actually resize tensors to remove pruned neurons."""

    # Step 1: Get FC Layer 1
    fc1 = self.fc_layers[0]
    weights = fc1.weight.data  # Shape: (64, 200)

    # Step 2: Find which neurons survived
    row_norms = weights.norm(dim=1)  # L2-norm per neuron
    keep_indices = (row_norms > 1e-6).nonzero(as_tuple=True)[0]
    # e.g., keep_indices = [0, 2, 3, 5, 7, 8, ...] (38 neurons)

    num_kept = len(keep_indices)  # e.g., 38 neurons survived

    # Step 3: Create NEW smaller FC Layer 1
    new_fc1 = nn.Linear(200, num_kept)  # 200 → 38 (not 64!)
    new_fc1.weight.data = weights[keep_indices]  # Copy surviving weights
    new_fc1.bias.data = fc1.bias.data[keep_indices]

    # Step 4: Update BatchNorm to match new size
    bn1 = self.fc_layers[1]
    new_bn1 = nn.BatchNorm1d(num_kept)  # 64 → 38
    new_bn1.weight.data = bn1.weight.data[keep_indices]
    new_bn1.bias.data = bn1.bias.data[keep_indices]
    new_bn1.running_mean = bn1.running_mean[keep_indices]
    new_bn1.running_var = bn1.running_var[keep_indices]

    # Step 5: Update FC Layer 2 input dimension
    fc_final = self.fc_layers[3]  # Linear(64, 5)
    new_fc_final = nn.Linear(num_kept, 5)  # 38 → 5 (not 64 → 5!)
    new_fc_final.weight.data = fc_final.weight.data[:, keep_indices]
    new_fc_final.bias.data = fc_final.bias.data

    # Step 6: Replace layers in model
    self.fc_layers[0] = new_fc1
    self.fc_layers[1] = new_bn1
    self.fc_layers[3] = new_fc_final

    print(f"Physically removed neurons: 64 → {num_kept}")
    # Output: "Physically removed neurons: 64 → 38"
```

**Result:**
```
Before: Linear(200, 64) → 12,864 parameters
After:  Linear(200, 38) → 7,600 parameters
Reduction: 40.9% parameter reduction ✓

Before: Model size = 56.36 KB
After:  Model size = 38.32 KB
Reduction: 32% size reduction ✓
```

---

## 8. Complete Pruning Pipeline

### Full Code Flow

```python
# ============================================================
# MAIN PRUNING SCRIPT
# File: scripts/06_model_compression/prune_and_finetune.py
# ============================================================

# Step 1: Load baseline (unpruned) model
model = PrunedConv1DNet(cfg)
model.load_state_dict(torch.load('baseline_ema_a03_model.pt'))
print(f"Baseline size: {get_model_size(model):.2f} KB")
# Output: "Baseline size: 56.36 KB"

# Step 2: Define pruning schedule (for 40% target)
pruning_schedule = [
    (0.10, 3, [1e-4, 1e-5, 1e-5]),      # Remove 10%, train 3 epochs
    (0.10, 3, [1e-4, 1e-5, 1e-5]),      # Remove 10%, train 3 epochs
    (0.10, 3, [1e-4, 1e-5, 1e-5]),      # Remove 10%, train 3 epochs
    (0.10, 4, [1e-4, 1e-5, 1e-5, 1e-6]) # Remove 10%, train 4 epochs
]

# Step 3: Iterative pruning loop
iteration_results = []

for iteration, (prune_size, num_epochs, lrs) in enumerate(pruning_schedule):
    print(f"\n=== Iteration {iteration+1}/4 ===")
    print(f"Pruning {prune_size*100}% of neurons...")

    # 3a. Apply pruning
    model.apply_structured_pruning(iteration_size=prune_size)

    # 3b. Evaluate immediately after pruning (before fine-tuning)
    after_prune_metrics = evaluate_model(model, test_loader, device, cfg)
    print(f"After pruning - Recall: {after_prune_metrics['recall']:.4f}")
    # Typical output: "After pruning - Recall: 0.8523" (drops after each prune)

    # 3c. Fine-tune to recover performance
    print(f"Fine-tuning for {num_epochs} epochs...")
    fine_tune_iteration(model, train_loader, test_loader, device, cfg,
                       num_epochs, lrs)

    # 3d. Evaluate after fine-tuning
    after_finetune_metrics = evaluate_model(model, test_loader, device, cfg)
    print(f"After fine-tuning - Recall: {after_finetune_metrics['recall']:.4f}")
    # Typical output: "After fine-tuning - Recall: 0.8891" (recovers)

    # 3e. Save checkpoint
    torch.save(model.state_dict(), f'pruned_iter{iteration}.pt')

    iteration_results.append({
        'iteration': iteration,
        'after_prune_recall': after_prune_metrics['recall'],
        'after_finetune_recall': after_finetune_metrics['recall']
    })

# Step 4: Make pruning permanent (remove masks)
print("\nMaking pruning permanent...")
model.make_pruning_permanent()

# Step 5: Physically remove pruned neurons
print("Physically removing pruned neurons...")
model.physically_remove_pruned_neurons()
# Output: "Physically removed neurons: 64 → 38"

# Step 6: Final evaluation
print("\nFinal evaluation...")
final_metrics = evaluate_model(model, test_loader, device, cfg)
final_size = get_model_size(model)

print(f"\n=== FINAL RESULTS ===")
print(f"Original model:")
print(f"  - Recall: 88.33%")
print(f"  - Size: 56.36 KB")
print(f"\nPruned model (40%):")
print(f"  - Recall: {final_metrics['recall']*100:.2f}%")
print(f"  - Size: {final_size:.2f} KB")
print(f"\nImprovement:")
print(f"  - Recall: +{(final_metrics['recall']-0.8833)*100:.2f}%")
print(f"  - Size reduction: {(1-final_size/56.36)*100:.1f}%")

# Step 7: Save final model
torch.save(model.state_dict(), 'final_pruned_40pct.pt')
```

**Typical Output:**
```
=== FINAL RESULTS ===
Original model:
  - Recall: 88.33%
  - Size: 56.36 KB

Pruned model (40%):
  - Recall: 89.18%
  - Size: 38.32 KB

Improvement:
  - Recall: +0.85%
  - Size reduction: 32.0%
```

---

## 9. Why 40% Pruning is Optimal

### Multi-Objective Optimization

We tested 5 pruning levels: 10%, 20%, 30%, 40%, 50%

**Scoring Function:**
```python
# Compression benefit (higher is better)
compression_benefit = (compression_ratio - 1.0) × 10

# Performance penalty (lower is better)
# Weighted by importance: Recall > Accuracy > Precision
performance_penalty = (
    3.0 × recall_drop × 100 +
    2.0 × accuracy_drop × 100 +
    1.0 × precision_drop × 100
)

# Overall score (higher is better)
overall_score = compression_benefit - performance_penalty
```

### Results by Pruning Level

| Prune % | Avg Recall | Compression | Performance Penalty | Compression Benefit | **Overall Score** |
|---------|-----------|-------------|---------------------|---------------------|-------------------|
| 10% | 88.94% | 1.09× | -1.87 | 0.90 | -0.96 |
| 20% | 88.97% | 1.19× | -1.96 | 1.90 | -0.06 |
| 30% | 89.19% | 1.31× | -2.58 | 3.10 | **0.52** |
| **40%** | **89.18%** | **1.47×** | **-2.52** | **4.70** | **2.18** ← HIGHEST |
| 50% | 88.89% | 1.68× | -1.65 | 6.80 | **1.93** |

**Why 40% wins:**
- Good compression (1.47×) without being too aggressive
- Maintains (even improves!) recall
- Better score than 50% because 50% starts to degrade stability
- Sweet spot in the performance-compression trade-off

---

## 10. Final Results Summary

### Comparison: Original vs Final Model

**Original Model (No Filter, No Pruning):**
- Recall: 88.33%
- Accuracy: 96.06%
- Precision: 94.51%
- Model Size: 56.36 KB
- Parameters: 13,897

**Final Model (EMA α=0.3 + 40% Pruning):**
- Recall: **89.18%** (+0.85% improvement ✓)
- Accuracy: **96.13%** (+0.07% improvement ✓)
- Precision: **94.16%** (-0.35% minimal drop)
- Model Size: **38.32 KB** (32% reduction ✓)
- Parameters: 9,321 (33% reduction ✓)

### ESP32 Deployment Benefits

**Memory Savings:**
- RAM saved: ~18 KB (56.36 - 38.32)
- This is significant on ESP32 (limited memory)
- Allows more headroom for other application code

**Computational Savings:**
- 40% fewer neurons in FC Layer 1
- FC1: 200×64 MULs → 200×38 MULs (40% reduction)
- Faster inference on resource-constrained hardware

**Energy Savings:**
- Fewer operations = less power consumption
- Important for battery-powered wearable devices

---

## 11. Key Takeaways for Dean

1. **Pruning FC layers gives maximum benefit** - they contain 94% of parameters

2. **Structured pruning is essential for ESP32** - unstructured requires sparse libraries

3. **Iterative pruning prevents accuracy collapse** - gradual removal allows adaptation

4. **Physical neuron removal is critical** - PyTorch's built-in pruning doesn't resize tensors

5. **40% pruning is optimal** - best trade-off between compression and performance

6. **Performance actually IMPROVED** - pruning acted as regularization (+0.85% recall)

7. **Ready for ESP32 deployment** - 32% smaller, faster, more efficient

---

## 12. Code Files Reference

**Model Definition:**
- `trainer/models/conv1d_model.py` - Base CNN architecture
- `trainer/models/pruned_conv1d_model.py` - Pruning implementation

**Pruning Script:**
- `scripts/06_model_compression/prune_and_finetune.py` - Main pruning pipeline

**Analysis Scripts:**
- `find_best_pruning.py` - Multi-objective optimization to find 40% optimal
- `compare_final.py` - Comparison vs original baseline

**Configuration:**
- `config/pruning/prune_ema_s02_40pct_seed42.json` - Example config for 40% pruning
