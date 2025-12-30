#!/bin/bash

# Test script to verify ConfigManager fix is working
# Run this BEFORE submitting the full experiments

echo "========================================================================"
echo "TESTING CONFIGMANAGER FIX"
echo "========================================================================"
echo ""
echo "This will run ONE experiment to verify filters are being applied correctly."
echo ""

cd ~/ankleband

# Check if config exists
# if [ ! -f "config/filter_redo/baseline_s02.json" ]; then
#     echo "❌ ERROR: Config files not found!"
#     echo "   Run: python scripts/filter_redo/generate_filter_redo_configs.py"
#     exit 1
# fi

# echo "Step 1: Testing Baseline (no filter)"
# echo "--------------------------------------------------------------------"
# python trainer/train_conv.py --json config/filter_redo/baseline_s02.json --loo 2 > test_baseline.log 2>&1 &
# BASELINE_PID=$!

# echo "  Job started (PID: $BASELINE_PID)"
# echo "  Waiting 10 seconds for initialization..."
# sleep 10

# # Check log
# if grep -q "Applying butterworth filter" test_baseline.log; then
#     echo "  ❌ FAILED: Baseline should not apply any filter!"
#     echo "  ConfigManager fix may not be working"
#     kill $BASELINE_PID 2>/dev/null
#     exit 1
# elif grep -q "Loading dataset" test_baseline.log; then
#     echo "  ✓ Baseline looks good (no filter being applied)"
# else
#     echo "  ⚠ Cannot determine - check test_baseline.log manually"
# fi

echo ""
echo "Step 2: Testing Kalman filter"
echo "--------------------------------------------------------------------"

# Kill baseline job
kill $BASELINE_PID 2>/dev/null
sleep 2

python trainer/train_conv.py --json config/filter_redo/kalman_q0001_r0001_s02.json --loo 2 > test_kalman.log 2>&1 &
KALMAN_PID=$!

echo "  Job started (PID: $KALMAN_PID)"
echo "  Waiting 10 seconds for initialization..."
sleep 10

# Check log
if grep -q "Filter settings: Kalman Q=0.0001 R=0.0001" test_kalman.log; then
    echo "  ✓ SUCCESS: Kalman filter is being applied correctly!"
    KALMAN_OK=1
elif grep -q "Applying butterworth filter" test_kalman.log; then
    echo "  ❌ FAILED: Should be Kalman, but seeing Butterworth!"
    echo "  ConfigManager fix NOT working - check trainer/utils.py"
    kill $KALMAN_PID 2>/dev/null
    exit 1
else
    echo "  ⚠ Cannot determine - check test_kalman.log manually"
    KALMAN_OK=0
fi

# Kill kalman job
kill $KALMAN_PID 2>/dev/null

echo ""
echo "========================================================================"
echo "TEST RESULTS"
echo "========================================================================"

if [ "$KALMAN_OK" = "1" ]; then
    echo "✓ ConfigManager fix is working!"
    echo ""
    echo "You can now submit the full experiments:"
    echo "  Account 1: sbatch scripts/filter_redo/run_account1_filters.sh"
    echo "  Account 2: sbatch scripts/filter_redo/run_account2_filters.sh"
    echo ""
    echo "Monitor progress:"
    echo "  watch -n 30 'squeue -u \$USER'"
    echo ""
else
    echo "⚠ Test inconclusive - check logs manually:"
    echo "  test_baseline.log"
    echo "  test_kalman.log"
    echo ""
    echo "Look for lines like:"
    echo "  'Applying kalman filter to sensor data...'"
    echo "  'Filter settings: Kalman Q=0.0001 R=0.0001'"
fi

echo ""
echo "Cleaning up test logs..."
rm -f test_baseline.log test_kalman.log

echo "Done!"
