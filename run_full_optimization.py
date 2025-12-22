#!/usr/bin/env python3
"""
Full Two-Phase Filter Optimization Orchestrator

Runs complete optimization workflow:
1. Phase 1: Coarse search (139 configs)
2. Phase 2: Fine search around best regions (~66 configs)

Total runtime: ~30 minutes
"""

import sys
import time
from pathlib import Path

# Import phase scripts
import optimize_filters_phase1 as phase1
import optimize_filters_phase2 as phase2


def print_header(text: str):
    """Print formatted header."""
    print("\n" + "="*80)
    print(text.center(80))
    print("="*80 + "\n")


def main():
    """Run full two-phase optimization."""
    start_time = time.time()

    print_header("TWO-PHASE FILTER OPTIMIZATION")
    print("This will test 200+ filter configurations to find optimal settings")
    print("for prosthetic hand control using discrete window architecture.\n")
    print("Estimated runtime: ~30 minutes\n")

    input("Press Enter to start optimization...")

    # Phase 1: Coarse Search
    print_header("PHASE 1: COARSE SEARCH (139 configs)")
    phase1_start = time.time()

    try:
        phase1_results = phase1.run_phase1_optimization()
        phase1_time = time.time() - phase1_start

        print(f"\n[OK] Phase 1 completed in {phase1_time/60:.1f} minutes")

    except Exception as e:
        print(f"\n[ERROR] Phase 1 failed: {e}")
        return

    # Brief pause
    print("\nPreparing for Phase 2...")
    time.sleep(2)

    # Phase 2: Fine Search
    print_header("PHASE 2: FINE SEARCH (~66 configs)")
    phase2_start = time.time()

    try:
        phase2_results = phase2.run_phase2_optimization()
        phase2_time = time.time() - phase2_start

        print(f"\n[OK] Phase 2 completed in {phase2_time/60:.1f} minutes")

    except Exception as e:
        print(f"\n[ERROR] Phase 2 failed: {e}")
        return

    # Final summary
    total_time = time.time() - start_time

    print_header("OPTIMIZATION COMPLETE")
    print(f"Total runtime: {total_time/60:.1f} minutes")
    print(f"  Phase 1: {phase1_time/60:.1f} minutes")
    print(f"  Phase 2: {phase2_time/60:.1f} minutes")

    print("\nResults:")
    print(f"  Phase 1: outputs/optimization_phase1/")
    print(f"    - phase1_coarse_results.csv")
    print(f"    - phase1_summary.txt")
    print(f"    - Visualizations")

    print(f"\n  Phase 2: outputs/optimization_phase2/")
    print(f"    - phase2_fine_results.csv")
    print(f"    - phase2_summary.txt")
    print(f"    - Visualizations")

    # Load and display winner
    if phase2_results is not None:
        df_avg = phase2_results.groupby(['filter_type', 'config_label']).agg({
            'composite_score': 'mean',
            'snr_db': 'mean',
            'edge_sharpness': 'mean',
            'correlation': 'mean',
            'phase_delay_ms': 'mean'
        }).reset_index()

        winner = df_avg.nlargest(1, 'composite_score').iloc[0]

        print("\n" + "="*80)
        print("WINNING CONFIGURATION")
        print("="*80)
        print(f"\nFilter: {winner['filter_type']}")
        print(f"Config: {winner['config_label']}")
        print(f"\nPerformance:")
        print(f"  Composite Score: {winner['composite_score']:.1f}/100")
        print(f"  SNR: {winner['snr_db']:.1f} dB")
        print(f"  Edge Sharpness: {winner['edge_sharpness']:.3f}")
        print(f"  Correlation: {winner['correlation']:.3f}")
        print(f"  Phase Delay: {winner['phase_delay_ms']:.1f} ms")

        print("\nNext steps:")
        print("1. Review detailed results in phase2_summary.txt")
        print("2. (Optional) Train CNN on top 3 configs to validate")
        print("3. Implement winner in execute_imu_gestures.ino")

    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user.")
        sys.exit(1)
