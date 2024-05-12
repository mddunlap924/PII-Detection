#!/bin/bash
path_file=./training/train_chunks_cib.py
path_cfg_dir=./cfgs/training

# Run Python module 1 in the background
CUDA_VISIBLE_DEVICES=0 python $path_file --dir $path_cfg_dir --name cfg0.yaml &

# Run Python module 2 in the background
CUDA_VISIBLE_DEVICES=1 python $path_file --dir $path_cfg_dir --name cfg1.yaml &

# Wait for both processes to finish
wait

echo "Both Python modules have finished execution."
