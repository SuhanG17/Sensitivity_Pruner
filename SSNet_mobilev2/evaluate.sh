#!/bin/bash
compression_rate=0.5
model_dir_path='results_0/model'
channel_dim_path='results_0/logs/block_7/channel_dim.pth'
checkpoint_path='results_0/checkpoint'

# 4. evaluate
LOG=results_0/logs/evaluate.log
python evaluate.py \
	--model_dir_path ${model_dir_path} \
	--channel_dim_path ${channel_dim_path} \
	--checkpoint_path ${checkpoint_path} 2>&1 \
    --compression_rate ${compression_rate} | tee ${LOG}