#!/bin/bash
block=2 #1-7
compression_rate=0.4
model_dir_path='results_0/model'
log_dir_path='results_0/logs'

# 1. prune
# COMP_LOG=results/logs/compress_model_${layer}.log
LOG=${log_dir_path}/log_${block}.log
python main.py \
	--train_path 'data/imagenet_64_64/prune/train_loader' \
	--val_path 'data/imagenet_64_64/prune/val_loader' \
	--model_dir_path ${model_dir_path} \
	--log_dir_path ${log_dir_path} \
	--compression_rate ${compression_rate} \
	--block_id ${block} 2>&1 | tee ${LOG} 

