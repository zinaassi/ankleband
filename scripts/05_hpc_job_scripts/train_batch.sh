#!/bin/bash

# arrays for frequency experiment
# values2=(120 60 40 30 24 20 17 15 13 12) # append values
# values1=($(seq 1 10))

# array for sequence lengths
values1=($(seq 10 10 150))

# array for label percentage lengths
# values1=($(seq 0.1 0.1 1.0))

# array for number of subjects in training
# values1=($(seq 1 9))

# Iterate over x (integers from 1 to 10)
for i in "${!values1[@]}"; do
  x="${values1[$i]}"
  # z="${values2[$i]}"

  echo "Current number: $x"

  # Iterate over y (integers from 10 to 150)
  for y in $(seq 1 10); do

    # sequence length experiment
    python trainer/train_conv.py --append "$x" --loo "$y" & # Run in background

    # label percentage experiment
    # python trainer/train_conv.py --label_percentage "$x" --loo "$y" & # Run in background

    # number of subjects experiment
    # python trainer/train_conv.py --nst "$x" --loo "$y" & # Run in background

    # frequency experiment
    # python trainer/train_conv.py --step "$x" --append "$z" --loo "$y" & #

    # classification experiment
    # python trainer/compute_metrics.py --append "$x" --step "$z" --loo "$y" & # Run in background
  done

  # Wait for all background processes to finish
  wait

  echo "Finished processing x=$x"
done

echo "All combinations processed."