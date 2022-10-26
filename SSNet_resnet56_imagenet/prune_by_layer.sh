#!/bin/bash
gpus=4,5sh


layer=0
compression_rate=0.4
# 1. prune
# COMP_LOG=results/logs/compress_model_${layer}.log
LOG=results/logs/log_${layer}.log
python main.py \
	--train_path 'data/imagenet_64_64/prune/train_loader' \
	--val_path 'data/imagenet_64_64/prune/val_loader' \
	--compression_rate ${compression_rate} \
	--gpu_id ${gpus} \
	--group_id ${layer} 2>&1 | tee ${LOG}