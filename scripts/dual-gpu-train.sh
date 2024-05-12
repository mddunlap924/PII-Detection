#!/bin/bash

# Run Python module 1 in the background
python3 ./training/train_dual_gpu.py --dir ./cfgs/training --name cfg0.yaml

# Wait for both processes to finish
wait

echo "Both Python modules have finished execution."

# Run Python module 1 in the background
python3 ./training/train_dual_gpu.py --dir ./cfgs/training --name cfg1.yaml
