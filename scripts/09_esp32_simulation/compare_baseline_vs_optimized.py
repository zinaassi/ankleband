#!/usr/bin/env python3
"""
Baseline vs Optimized ESP32 Inference Comparison
=================================================
Compares the original FP32 model (no filtering, no pruning, no quantization)
against the fully optimized INT8 model (EMA filter + 40% pruning + QAT).

Generates:
  - Side-by-side comparison table
  - Bar chart visualizations for the report
  - CSV with all comparison data

Original (Baseline) model:
  Conv1D(6→10, k=3, s=3) → BN(200) → FC1(200→64) → BN(64) → FC2(64→5)
  13,897 parameters, all FP32, no preprocessing filter

Optimized model:
  EMA(α=0.3) → Conv1D(6→10) → BN(200) → FC1(200→42) → BN(42) → FC2(42→5)
  9,321 parameters, INT8 weights + FP32 BatchNorm, EMA preprocessing
"""

import os
import sys
import csv
import math
import numpy as np

# Try matplotlib - needed for visualization
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("Warning: matplotlib not available, skipping plots")


# ============================================================================
# ESP32 Hardware Constants (same for both models)
# ============================================================================

ESP32_CLOCK_MHZ = 240
ESP32_SRAM_KB = 520
ESP32_SUPPLY_V = 3.3
ESP32_ACTIVE_MA = 30.0    # CPU active @ 240 MHz, no WiFi/BT
ESP32_SLEEP_MA = 20.0     # Modem sleep between inferences

# Xtensa LX6 instruction cycles (from ISA Reference Manual)
C = {
    "int_mul": 1, "int_add": 1, "int_cmp": 1,
    "int_load": 1, "int_store": 1, "int_clamp": 2,
    "fp_mul": 1, "fp_add": 1, "fp_sub": 1,
    "fp_div": 6, "fp_sqrt": 12, "fp_round": 3, "fp_cmp": 1,
    "loop": 2,
}

INFERENCE_RATE_HZ = 100  # 200 Hz sensor / 2 subsampling


# ============================================================================
# Model Definitions
# ============================================================================

BASELINE = {
    "name": "Baseline (FP32)",
    "description": "Original model, no filter, no pruning, no quantization",
    "in_ch": 6, "in_len": 60,
    "kernel": 3, "stride": 3,
    "n_kernels": 10, "conv_out": 20, "flat": 200,
    "fc1_out": 64,        # NOT pruned
    "n_classes": 5,
    "has_ema": False,
    "is_int8": False,     # FP32 weights
    # Classification performance (from report)
    "accuracy": 96.06,
    "recall": 88.33,
    "precision": 94.51,
    "f1": 91.47,
}

OPTIMIZED = {
    "name": "Optimized (INT8)",
    "description": "EMA filter + 40% pruning + INT8 QAT",
    "in_ch": 6, "in_len": 60,
    "kernel": 3, "stride": 3,
    "n_kernels": 10, "conv_out": 20, "flat": 200,
    "fc1_out": 42,        # Pruned from 64
    "n_classes": 5,
    "has_ema": True,
    "is_int8": True,      # INT8 weights
    # Classification performance (from report)
    "accuracy": 95.67,
    "recall": 88.75,
    "precision": 93.95,
    "f1": 91.29,
}


# ============================================================================
# Memory Computation
# ============================================================================

def compute_memory(model):
    m = model
    bytes_per_weight = 1 if m["is_int8"] else 4

    # Weights
    conv_w = m["n_kernels"] * m["in_ch"] * m["kernel"] * bytes_per_weight
    fc1_w  = m["fc1_out"] * m["flat"] * bytes_per_weight
    fc2_w  = m["n_classes"] * m["fc1_out"] * bytes_per_weight
    weights = conv_w + fc1_w + fc2_w

    # Biases (always FP32)
    conv_b = m["n_kernels"] * 4
    fc1_b  = m["fc1_out"] * 4
    fc2_b  = m["n_classes"] * 4
    biases = conv_b + fc1_b + fc2_b

    # BatchNorm (always FP32: weight, bias, running_mean, running_var)
    bn1 = m["flat"] * 4 * 4        # 200 features
    bn2 = m["fc1_out"] * 4 * 4     # 64 or 42 features
    batchnorm = bn1 + bn2

    # Quantization params (INT8 only)
    quant = 3 * 5 if m["is_int8"] else 0

    model_total = weights + biases + batchnorm + quant

    # Runtime buffers
    # FP32 buffers for intermediate results
    buf_flat = m["flat"] * 4                    # conv output / BN1
    buf_fc1  = m["fc1_out"] * 4                 # FC1 output / BN2
    buf_cls  = m["n_classes"] * 4               # logits
    # INT8 buffers (only for INT8 model)
    int8_flat = m["flat"] if m["is_int8"] else 0
    int8_fc1  = m["fc1_out"] if m["is_int8"] else 0
    # Conv input quantization buffer (INT8 only, stack)
    conv_q_buf = (m["in_ch"] * m["in_len"]) if m["is_int8"] else 0
    static_bufs = buf_flat + buf_fc1 + buf_cls + int8_flat + int8_fc1 + conv_q_buf

    # Input buffer
    input_buf = m["in_ch"] * m["in_len"] * 4

    # EMA state
    ema = (m["in_ch"] * 5) if m["has_ema"] else 0  # 6 floats + 6 bools

    # Object overhead
    obj = 45 if m["is_int8"] else 20

    runtime = static_bufs + input_buf + ema + obj

    # Code estimate
    code = 4500 if m["is_int8"] else 3500  # INT8 engine slightly larger

    total = model_total + runtime + code

    return {
        "weights": weights,
        "biases": biases,
        "batchnorm": batchnorm,
        "quant_params": quant,
        "model_total": model_total,
        "runtime": runtime,
        "code": code,
        "total": total,
        "param_count": (conv_w + fc1_w + fc2_w) // bytes_per_weight + \
                       (conv_b + fc1_b + fc2_b) // 4,
        # Breakdown for display
        "conv_w": conv_w, "fc1_w": fc1_w, "fc2_w": fc2_w,
    }


# ============================================================================
# Cycle Counting
# ============================================================================

def count_cycles(model):
    """Count inference cycles for either FP32 or INT8 model."""
    m = model
    stages = []

    # ---- EMA filter ----
    if m["has_ema"]:
        iters = m["in_len"] * m["in_ch"]  # 360
        per = 2 * C["fp_mul"] + C["fp_add"] + C["int_store"] + C["loop"]
        stages.append(("EMA filter", iters * per))
    else:
        stages.append(("EMA filter", 0))

    if m["is_int8"]:
        # ---- INT8 pipeline (with quantize/dequantize steps) ----
        n_in = m["in_ch"] * m["in_len"]  # 360
        q_per = C["fp_mul"] + C["fp_round"] + C["int_clamp"] + C["int_store"] + C["loop"]

        # Quantize conv input
        stages.append(("Quantize conv input", n_in * q_per))

        # Conv1D INT8
        inner = m["in_ch"] * m["kernel"]  # 18
        mac_per = 2 * C["int_load"] + C["int_mul"] + C["int_add"] + C["loop"]
        requant = C["fp_mul"] + C["fp_round"] + C["int_clamp"] + C["int_store"]
        conv_per = inner * mac_per + requant + C["loop"]
        conv = m["n_kernels"] * (m["conv_out"] * conv_per + C["loop"])
        stages.append(("Conv1D", conv))

        # Dequantize conv output
        dq_per = C["int_add"] + C["fp_mul"] + C["int_store"] + C["loop"]
        stages.append(("Dequant conv", m["flat"] * dq_per))

        # BN1
        bn_per = (C["fp_sub"] + C["fp_add"] + C["fp_sqrt"] + C["fp_div"]
                  + C["fp_mul"] + C["fp_add"] + C["int_store"] + C["loop"])
        stages.append(("BatchNorm1", m["flat"] * bn_per))

        # ReLU1
        relu_per = C["fp_cmp"] + C["int_store"] + C["loop"]
        stages.append(("ReLU1", m["flat"] * relu_per))

        # Quantize FC1 input
        stages.append(("Quantize FC1", m["flat"] * q_per))

        # FC1 INT8
        fc1_per = m["flat"] * mac_per + requant + C["loop"]
        stages.append(("FC1", m["fc1_out"] * fc1_per))

        # Dequantize FC1 output
        stages.append(("Dequant FC1", m["fc1_out"] * dq_per))

        # BN2
        stages.append(("BatchNorm2", m["fc1_out"] * bn_per))

        # ReLU2
        stages.append(("ReLU2", m["fc1_out"] * relu_per))

        # Quantize FC2 input
        stages.append(("Quantize FC2", m["fc1_out"] * q_per))

        # FC2 INT8
        fc2_mac_per = 2 * C["int_load"] + C["int_mul"] + C["int_add"] + C["loop"]
        fc2_dq = C["fp_mul"] + C["int_store"]
        fc2_per = m["fc1_out"] * fc2_mac_per + fc2_dq + C["loop"]
        stages.append(("FC2", m["n_classes"] * fc2_per))

    else:
        # ---- FP32 pipeline (no quantize/dequantize needed) ----

        # Conv1D FP32
        inner = m["in_ch"] * m["kernel"]  # 18
        # Per MAC: 2 loads + 1 fp_mul + 1 fp_add + loop
        mac_per = 2 * C["int_load"] + C["fp_mul"] + C["fp_add"] + C["loop"]
        conv_per = inner * mac_per + C["int_store"] + C["loop"]
        conv = m["n_kernels"] * (m["conv_out"] * conv_per + C["loop"])
        stages.append(("Conv1D", conv))

        # BN1
        bn_per = (C["fp_sub"] + C["fp_add"] + C["fp_sqrt"] + C["fp_div"]
                  + C["fp_mul"] + C["fp_add"] + C["int_store"] + C["loop"])
        stages.append(("BatchNorm1", m["flat"] * bn_per))

        # ReLU1
        relu_per = C["fp_cmp"] + C["int_store"] + C["loop"]
        stages.append(("ReLU1", m["flat"] * relu_per))

        # FC1 FP32: 64 neurons x 200 inputs
        fc1_mac_per = 2 * C["int_load"] + C["fp_mul"] + C["fp_add"] + C["loop"]
        fc1_per = m["flat"] * fc1_mac_per + C["int_store"] + C["loop"]
        stages.append(("FC1", m["fc1_out"] * fc1_per))

        # BN2
        stages.append(("BatchNorm2", m["fc1_out"] * bn_per))

        # ReLU2
        stages.append(("ReLU2", m["fc1_out"] * relu_per))

        # FC2 FP32: 5 x 64
        fc2_per = m["fc1_out"] * fc1_mac_per + C["int_store"] + C["loop"]
        stages.append(("FC2", m["n_classes"] * fc2_per))

    # Argmax
    stages.append(("Argmax", m["n_classes"] * (C["fp_cmp"] + C["int_store"] + C["loop"])))

    return stages


# ============================================================================
# Derived Metrics
# ============================================================================

def compute_metrics(model):
    """Compute all metrics for a model configuration."""
    mem = compute_memory(model)
    stages = count_cycles(model)
    total_cycles = sum(cyc for _, cyc in stages)

    time_us = total_cycles / ESP32_CLOCK_MHZ
    time_ms = time_us / 1000
    throughput = 1e6 / time_us if time_us > 0 else 0
    energy_uj = ESP32_ACTIVE_MA * 1e-3 * ESP32_SUPPLY_V * time_us

    active_frac = time_us * INFERENCE_RATE_HZ * 1e-6
    duty_pct = active_frac * 100
    avg_ma = ESP32_ACTIVE_MA * active_frac + ESP32_SLEEP_MA * (1 - active_frac)
    avg_mw = avg_ma * ESP32_SUPPLY_V

    return {
        "name": model["name"],
        "mem": mem,
        "stages": stages,
        "total_cycles": total_cycles,
        "time_us": time_us,
        "time_ms": time_ms,
        "throughput": throughput,
        "energy_uj": energy_uj,
        "duty_pct": duty_pct,
        "avg_ma": avg_ma,
        "avg_mw": avg_mw,
        # Classification
        "accuracy": model["accuracy"],
        "recall": model["recall"],
        "precision": model["precision"],
        "f1": model["f1"],
    }


# ============================================================================
# Report Printing
# ============================================================================

def print_comparison(base, opt):
    W = 76
    sep = "=" * W
    sub = "-" * W

    print(f"\n{sep}")
    print("    BASELINE vs OPTIMIZED: ESP32 DEPLOYMENT COMPARISON")
    print(sep)

    # ---- Memory comparison ----
    print(f"\n{sub}")
    print("  1. MEMORY COMPARISON")
    print(sub)

    print(f"\n  {'Component':<36} {'Baseline':>12} {'Optimized':>12} {'Reduction':>12}")
    print(f"  {'~'*36} {'~'*12} {'~'*12} {'~'*12}")

    bm, om = base["mem"], opt["mem"]
    rows = [
        ("Weights", bm["weights"], om["weights"]),
        ("  Conv1D weights", bm["conv_w"], om["conv_w"]),
        ("  FC1 weights", bm["fc1_w"], om["fc1_w"]),
        ("  FC2 weights", bm["fc2_w"], om["fc2_w"]),
        ("Biases (FP32)", bm["biases"], om["biases"]),
        ("BatchNorm params (FP32)", bm["batchnorm"], om["batchnorm"]),
        ("Quantization params", bm["quant_params"], om["quant_params"]),
        ("Model subtotal", bm["model_total"], om["model_total"]),
        ("Runtime buffers", bm["runtime"], om["runtime"]),
        ("Code (.text est.)", bm["code"], om["code"]),
        ("TOTAL FOOTPRINT", bm["total"], om["total"]),
    ]

    for label, bv, ov in rows:
        if bv > 0:
            red = f"-{(1 - ov/bv)*100:.1f}%"
        else:
            red = "N/A"
        if label.startswith("  "):
            print(f"  {label:<36} {bv:>9,} B  {ov:>9,} B  {red:>12}")
        elif label in ("Model subtotal", "TOTAL FOOTPRINT"):
            print(f"  {'':-<36} {'':-<12} {'':-<12} {'':-<12}")
            bkb = f"{bv/1024:.2f} KB"
            okb = f"{ov/1024:.2f} KB"
            print(f"  {label:<36} {bkb:>12} {okb:>12} {red:>12}")
        else:
            print(f"  {label:<36} {bv:>9,} B  {ov:>9,} B  {red:>12}")

    # ---- Cycle comparison ----
    print(f"\n{sub}")
    print("  2. INFERENCE CYCLE COMPARISON (@ 240 MHz)")
    print(sub)

    print(f"\n  {'Metric':<36} {'Baseline':>14} {'Optimized':>14} {'Speedup':>10}")
    print(f"  {'~'*36} {'~'*14} {'~'*14} {'~'*10}")

    metrics = [
        ("Total CPU cycles", f"{base['total_cycles']:,}", f"{opt['total_cycles']:,}",
         f"{base['total_cycles']/opt['total_cycles']:.2f}x"),
        ("Inference latency", f"{base['time_us']:.0f} us", f"{opt['time_us']:.0f} us",
         f"{base['time_us']/opt['time_us']:.2f}x"),
        ("Inference latency", f"{base['time_ms']:.3f} ms", f"{opt['time_ms']:.3f} ms", ""),
        ("Throughput", f"{base['throughput']:,.0f} /s", f"{opt['throughput']:,.0f} /s",
         f"{opt['throughput']/base['throughput']:.2f}x"),
    ]
    for label, bv, ov, sp in metrics:
        print(f"  {label:<36} {bv:>14} {ov:>14} {sp:>10}")

    # ---- Power comparison ----
    print(f"\n{sub}")
    print("  3. POWER / ENERGY COMPARISON (@ 240 MHz, 3.3V, 100 Hz)")
    print(sub)

    print(f"\n  {'Metric':<36} {'Baseline':>14} {'Optimized':>14} {'Reduction':>10}")
    print(f"  {'~'*36} {'~'*14} {'~'*14} {'~'*10}")

    pmetrics = [
        ("Energy per inference",
         f"{base['energy_uj']:.2f} uJ", f"{opt['energy_uj']:.2f} uJ",
         f"-{(1-opt['energy_uj']/base['energy_uj'])*100:.1f}%"),
        ("CPU duty cycle @ 100 Hz",
         f"{base['duty_pct']:.2f}%", f"{opt['duty_pct']:.2f}%", ""),
        ("Average current @ 100 Hz",
         f"{base['avg_ma']:.2f} mA", f"{opt['avg_ma']:.2f} mA",
         f"-{(1-opt['avg_ma']/base['avg_ma'])*100:.1f}%"),
        ("Average power @ 100 Hz",
         f"{base['avg_mw']:.2f} mW", f"{opt['avg_mw']:.2f} mW",
         f"-{(1-opt['avg_mw']/base['avg_mw'])*100:.1f}%"),
    ]
    for label, bv, ov, red in pmetrics:
        print(f"  {label:<36} {bv:>14} {ov:>14} {red:>10}")

    # ---- Classification performance ----
    print(f"\n{sub}")
    print("  4. CLASSIFICATION PERFORMANCE")
    print(sub)

    print(f"\n  {'Metric':<36} {'Baseline':>14} {'Optimized':>14} {'Delta':>10}")
    print(f"  {'~'*36} {'~'*14} {'~'*14} {'~'*10}")
    for metric in ["accuracy", "recall", "precision", "f1"]:
        bv = base[metric]
        ov = opt[metric]
        delta = ov - bv
        sign = "+" if delta >= 0 else ""
        print(f"  {metric.capitalize():<36} {bv:>13.2f}% {ov:>13.2f}% {sign}{delta:>8.2f}pp")

    # ---- Summary ----
    print(f"\n{sub}")
    print("  5. OPTIMIZATION SUMMARY")
    print(sub)

    mem_red = (1 - om["total"] / bm["total"]) * 100
    speed_up = base["time_us"] / opt["time_us"]
    energy_red = (1 - opt["energy_uj"] / base["energy_uj"]) * 100

    print(f"""
  The complete optimization pipeline (EMA filtering + 40% structured
  pruning + INT8 quantization) achieves:

    Memory reduction:     {mem_red:.1f}%  ({bm['total']/1024:.1f} KB -> {om['total']/1024:.1f} KB)
    Inference speedup:    {speed_up:.1f}x   ({base['time_ms']:.3f} ms -> {opt['time_ms']:.3f} ms)
    Energy reduction:     {energy_red:.1f}%  ({base['energy_uj']:.1f} uJ -> {opt['energy_uj']:.1f} uJ)
    Recall change:        {opt['recall']-base['recall']:+.2f} pp  ({base['recall']:.2f}% -> {opt['recall']:.2f}%)

  These gains are achieved while maintaining classification performance
  within acceptable bounds (F1-score: {base['f1']:.2f}% -> {opt['f1']:.2f}%, delta = {opt['f1']-base['f1']:.2f} pp).
""")
    print(sep)


# ============================================================================
# Visualization
# ============================================================================

def create_visualizations(base, opt, output_dir):
    """Create comparison bar charts for the report."""
    if not HAS_MPL:
        print("Skipping visualizations (matplotlib not available)")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Color scheme
    COL_BASE = "#5B9BD5"   # Blue
    COL_OPT  = "#70AD47"   # Green

    # =========================================
    # Figure 1: Main comparison (4 panels)
    # =========================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Baseline vs Optimized Model: ESP32 Deployment Comparison",
                 fontsize=14, fontweight="bold", y=0.98)

    bar_width = 0.35
    x = np.array([0])

    # --- Panel 1: Memory ---
    ax = axes[0, 0]
    bv = base["mem"]["total"] / 1024
    ov = opt["mem"]["total"] / 1024
    bars1 = ax.bar(x - bar_width/2, [bv], bar_width, label="Baseline (FP32)", color=COL_BASE, edgecolor="white")
    bars2 = ax.bar(x + bar_width/2, [ov], bar_width, label="Optimized (INT8)", color=COL_OPT, edgecolor="white")
    ax.bar_label(bars1, fmt="%.1f KB", padding=3, fontsize=10, fontweight="bold")
    ax.bar_label(bars2, fmt="%.1f KB", padding=3, fontsize=10, fontweight="bold")
    ax.set_ylabel("Memory (KB)", fontsize=11)
    ax.set_title("Total SRAM Footprint", fontsize=12, fontweight="bold")
    ax.set_xticks([])
    ax.legend(fontsize=9)
    red = (1 - ov/bv) * 100
    ax.annotate(f"-{red:.1f}%", xy=(0, ov), fontsize=13, fontweight="bold",
                color="#C00000", ha="center", va="bottom",
                xytext=(0.3, ov + bv*0.08))

    # --- Panel 2: Inference Latency ---
    ax = axes[0, 1]
    bv = base["time_ms"]
    ov = opt["time_ms"]
    bars1 = ax.bar(x - bar_width/2, [bv], bar_width, color=COL_BASE, edgecolor="white")
    bars2 = ax.bar(x + bar_width/2, [ov], bar_width, color=COL_OPT, edgecolor="white")
    ax.bar_label(bars1, fmt="%.3f ms", padding=3, fontsize=10, fontweight="bold")
    ax.bar_label(bars2, fmt="%.3f ms", padding=3, fontsize=10, fontweight="bold")
    ax.set_ylabel("Latency (ms)", fontsize=11)
    ax.set_title("Inference Latency @ 240 MHz", fontsize=12, fontweight="bold")
    ax.set_xticks([])
    speedup = bv / ov
    ax.annotate(f"{speedup:.1f}x faster", xy=(0, ov), fontsize=13, fontweight="bold",
                color="#C00000", ha="center", va="bottom",
                xytext=(0.3, ov + bv*0.08))

    # --- Panel 3: Energy per Inference ---
    ax = axes[1, 0]
    bv = base["energy_uj"]
    ov = opt["energy_uj"]
    bars1 = ax.bar(x - bar_width/2, [bv], bar_width, color=COL_BASE, edgecolor="white")
    bars2 = ax.bar(x + bar_width/2, [ov], bar_width, color=COL_OPT, edgecolor="white")
    ax.bar_label(bars1, fmt="%.1f uJ", padding=3, fontsize=10, fontweight="bold")
    ax.bar_label(bars2, fmt="%.1f uJ", padding=3, fontsize=10, fontweight="bold")
    ax.set_ylabel("Energy (uJ)", fontsize=11)
    ax.set_title("Energy per Inference", fontsize=12, fontweight="bold")
    ax.set_xticks([])
    red = (1 - ov/bv) * 100
    ax.annotate(f"-{red:.1f}%", xy=(0, ov), fontsize=13, fontweight="bold",
                color="#C00000", ha="center", va="bottom",
                xytext=(0.3, ov + bv*0.08))

    # --- Panel 4: Classification Metrics ---
    ax = axes[1, 1]
    metrics_names = ["Accuracy", "Recall", "Precision", "F1-Score"]
    base_vals = [base["accuracy"], base["recall"], base["precision"], base["f1"]]
    opt_vals  = [opt["accuracy"], opt["recall"], opt["precision"], opt["f1"]]
    x4 = np.arange(len(metrics_names))
    bars1 = ax.bar(x4 - bar_width/2, base_vals, bar_width, label="Baseline", color=COL_BASE, edgecolor="white")
    bars2 = ax.bar(x4 + bar_width/2, opt_vals, bar_width, label="Optimized", color=COL_OPT, edgecolor="white")
    ax.bar_label(bars1, fmt="%.1f%%", padding=3, fontsize=8, fontweight="bold")
    ax.bar_label(bars2, fmt="%.1f%%", padding=3, fontsize=8, fontweight="bold")
    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_title("Classification Performance", fontsize=12, fontweight="bold")
    ax.set_xticks(x4)
    ax.set_xticklabels(metrics_names, fontsize=10)
    ax.set_ylim(85, 100)
    ax.legend(fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(output_dir, "baseline_vs_optimized_comparison.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # =========================================
    # Figure 2: Memory breakdown stacked bar
    # =========================================
    fig, ax = plt.subplots(figsize=(8, 5))

    categories = ["Baseline (FP32)", "Optimized (INT8)"]
    weights_vals  = [base["mem"]["weights"]/1024, opt["mem"]["weights"]/1024]
    biases_vals   = [base["mem"]["biases"]/1024, opt["mem"]["biases"]/1024]
    bn_vals       = [base["mem"]["batchnorm"]/1024, opt["mem"]["batchnorm"]/1024]
    runtime_vals  = [base["mem"]["runtime"]/1024, opt["mem"]["runtime"]/1024]
    code_vals     = [base["mem"]["code"]/1024, opt["mem"]["code"]/1024]

    x2 = np.arange(len(categories))
    w2 = 0.45

    p1 = ax.bar(x2, weights_vals, w2, label="Weights", color="#4472C4")
    p2 = ax.bar(x2, biases_vals, w2, bottom=weights_vals, label="Biases", color="#ED7D31")
    bottom2 = [a+b for a,b in zip(weights_vals, biases_vals)]
    p3 = ax.bar(x2, bn_vals, w2, bottom=bottom2, label="BatchNorm", color="#A5A5A5")
    bottom3 = [a+b for a,b in zip(bottom2, bn_vals)]
    p4 = ax.bar(x2, runtime_vals, w2, bottom=bottom3, label="Runtime buffers", color="#FFC000")
    bottom4 = [a+b for a,b in zip(bottom3, runtime_vals)]
    p5 = ax.bar(x2, code_vals, w2, bottom=bottom4, label="Code", color="#5B9BD5")

    totals = [sum(x) for x in zip(weights_vals, biases_vals, bn_vals, runtime_vals, code_vals)]
    for i, t in enumerate(totals):
        ax.text(i, t + 0.5, f"{t:.1f} KB", ha="center", fontweight="bold", fontsize=11)

    ax.set_ylabel("Memory (KB)", fontsize=12)
    ax.set_title("Memory Breakdown: Baseline vs Optimized", fontsize=13, fontweight="bold")
    ax.set_xticks(x2)
    ax.set_xticklabels(categories, fontsize=11)
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, "memory_breakdown_comparison.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # =========================================
    # Figure 3: Cycle breakdown by stage
    # =========================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Inference Cycle Distribution by Stage", fontsize=14, fontweight="bold")

    for ax, data, title, color_base in [(ax1, base, "Baseline (FP32)", "#4472C4"),
                                         (ax2, opt, "Optimized (INT8)", "#70AD47")]:
        stages = [(name, cyc) for name, cyc in data["stages"] if cyc > 0]
        labels = [s[0] for s in stages]
        values = [s[1] for s in stages]
        total = sum(values)
        pcts = [v/total*100 for v in values]

        # Sort by size for pie chart readability
        sorted_data = sorted(zip(labels, values, pcts), key=lambda x: -x[1])
        labels_s = [d[0] for d in sorted_data]
        values_s = [d[1] for d in sorted_data]
        pcts_s = [d[2] for d in sorted_data]

        # Color gradient
        n = len(labels_s)
        colors = plt.cm.Blues(np.linspace(0.3, 0.85, n)) if "Baseline" in title \
                 else plt.cm.Greens(np.linspace(0.3, 0.85, n))

        def make_label(pct):
            return f"{pct:.1f}%" if pct >= 3 else ""

        wedges, texts, autotexts = ax.pie(
            values_s, labels=None, autopct=make_label,
            colors=colors, startangle=90,
            pctdistance=0.75, textprops={"fontsize": 9}
        )

        ax.legend(wedges, [f"{l} ({p:.1f}%)" for l, p in zip(labels_s, pcts_s)],
                  loc="center left", bbox_to_anchor=(-0.3, 0.5), fontsize=8)
        ax.set_title(f"{title}\n{total:,} cycles ({total/ESP32_CLOCK_MHZ:.0f} us)",
                     fontsize=11, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(output_dir, "cycle_distribution_comparison.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # =========================================
    # Figure 4: Pipeline optimization summary
    # =========================================
    fig, ax = plt.subplots(figsize=(10, 5))

    stages_labels = ["Original\n(FP32)", "+ EMA Filter\n(FP32)", "+ 40% Pruning\n(FP32)", "+ INT8 QAT\n(Optimized)"]
    # Compute intermediate stages
    # Stage 1: baseline
    # Stage 2: baseline + EMA (same model, just adds filter)
    # Stage 3: baseline + EMA + pruning (fc1: 64->42, still FP32)
    # Stage 4: baseline + EMA + pruning + INT8 (final)

    # Memory at each stage (model total)
    base_model_bytes = base["mem"]["model_total"]
    # With EMA: same model
    ema_model_bytes = base_model_bytes
    # With pruning (FP32): FC1 200->42 instead of 200->64
    pruned_fp32_fc1 = 42 * 200 * 4  # 33,600
    pruned_fp32_fc2 = 5 * 42 * 4    # 840
    pruned_fp32_bn2 = 42 * 4 * 4    # 672
    pruned_fp32_bias = 10*4 + 42*4 + 5*4  # 228
    pruned_fp32_conv = 10 * 6 * 3 * 4  # 720
    pruned_fp32_bn1 = 200 * 4 * 4  # 3200
    pruned_model = pruned_fp32_conv + pruned_fp32_fc1 + pruned_fp32_fc2 + \
                   pruned_fp32_bias + pruned_fp32_bn1 + pruned_fp32_bn2
    # With INT8
    int8_model = opt["mem"]["model_total"]

    mem_stages = [base_model_bytes/1024, ema_model_bytes/1024,
                  pruned_model/1024, int8_model/1024]

    colors_bar = ["#C00000", "#ED7D31", "#FFC000", "#70AD47"]
    bars = ax.bar(range(4), mem_stages, color=colors_bar, edgecolor="white", width=0.6)
    ax.bar_label(bars, [f"{v:.1f} KB" for v in mem_stages], padding=3,
                 fontsize=11, fontweight="bold")

    ax.set_xticks(range(4))
    ax.set_xticklabels(stages_labels, fontsize=10)
    ax.set_ylabel("Model Memory (KB)", fontsize=12)
    ax.set_title("Progressive Memory Reduction Through Optimization Pipeline",
                 fontsize=13, fontweight="bold")

    # Add arrows with reduction %
    for i in range(1, 4):
        if mem_stages[i] < mem_stages[i-1]:
            red = (1 - mem_stages[i]/mem_stages[i-1]) * 100
            mid_y = (mem_stages[i-1] + mem_stages[i]) / 2
            ax.annotate(f"-{red:.0f}%", xy=(i, mem_stages[i]),
                        xytext=(i-0.5, mid_y),
                        fontsize=10, fontweight="bold", color="#333333",
                        arrowprops=dict(arrowstyle="->", color="#333333"),
                        ha="center")

    plt.tight_layout()
    path = os.path.join(output_dir, "pipeline_memory_reduction.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
# CSV Export
# ============================================================================

def save_comparison_csv(base, opt, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "baseline_vs_optimized.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "Baseline (FP32)", "Optimized (INT8)", "Change", "Unit"])
        bm, om = base["mem"], opt["mem"]

        w.writerow(["Model memory", f"{bm['model_total']/1024:.2f}", f"{om['model_total']/1024:.2f}",
                     f"-{(1-om['model_total']/bm['model_total'])*100:.1f}%", "KB"])
        w.writerow(["Total SRAM footprint", f"{bm['total']/1024:.2f}", f"{om['total']/1024:.2f}",
                     f"-{(1-om['total']/bm['total'])*100:.1f}%", "KB"])
        w.writerow(["SRAM utilization", f"{bm['total']/bm['total']*100:.1f}",  # will fix
                     f"{om['total']/(ESP32_SRAM_KB*1024)*100:.1f}", "", "%"])
        w.writerow(["Inference cycles", f"{base['total_cycles']}", f"{opt['total_cycles']}",
                     f"{base['total_cycles']/opt['total_cycles']:.2f}x faster", "cycles"])
        w.writerow(["Inference latency", f"{base['time_us']:.0f}", f"{opt['time_us']:.0f}",
                     f"{base['time_us']/opt['time_us']:.2f}x faster", "us"])
        w.writerow(["Inference latency", f"{base['time_ms']:.3f}", f"{opt['time_ms']:.3f}",
                     "", "ms"])
        w.writerow(["Throughput @ 240 MHz", f"{base['throughput']:.0f}", f"{opt['throughput']:.0f}",
                     f"{opt['throughput']/base['throughput']:.2f}x", "inf/s"])
        w.writerow(["Energy per inference", f"{base['energy_uj']:.2f}", f"{opt['energy_uj']:.2f}",
                     f"-{(1-opt['energy_uj']/base['energy_uj'])*100:.1f}%", "uJ"])
        w.writerow(["Avg power @ 100 Hz", f"{base['avg_mw']:.2f}", f"{opt['avg_mw']:.2f}",
                     f"-{(1-opt['avg_mw']/base['avg_mw'])*100:.1f}%", "mW"])
        w.writerow(["Accuracy", f"{base['accuracy']:.2f}", f"{opt['accuracy']:.2f}",
                     f"{opt['accuracy']-base['accuracy']:+.2f} pp", "%"])
        w.writerow(["Recall", f"{base['recall']:.2f}", f"{opt['recall']:.2f}",
                     f"{opt['recall']-base['recall']:+.2f} pp", "%"])
        w.writerow(["Precision", f"{base['precision']:.2f}", f"{opt['precision']:.2f}",
                     f"{opt['precision']-base['precision']:+.2f} pp", "%"])
        w.writerow(["F1-Score", f"{base['f1']:.2f}", f"{opt['f1']:.2f}",
                     f"{opt['f1']-base['f1']:+.2f} pp", "%"])

    print(f"  Saved: {os.path.join(output_dir, 'baseline_vs_optimized.csv')}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    base = compute_metrics(BASELINE)
    opt  = compute_metrics(OPTIMIZED)

    print_comparison(base, opt)

    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "outputs", "esp32_simulation"
    )

    print("\nGenerating visualizations...")
    create_visualizations(base, opt, output_dir)

    print("\nSaving CSV...")
    save_comparison_csv(base, opt, output_dir)

    print("\nDone!")
