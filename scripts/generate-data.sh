#!/bin/bash

# Run Python module 1 in the background
CUDA_VISIBLE_DEVICES=0 python3 ./gen-data/ai-gen.py --dir ./gen-data/cfgs --name cfg0.yaml &

# Run Python module 2 in the background
CUDA_VISIBLE_DEVICES=1 python3 ./gen-data/ai-gen.py --dir ./gen-data/cfgs --name cfg1.yaml &

# Wait for both processes to finish
wait