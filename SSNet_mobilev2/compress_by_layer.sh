#!/bin/bash
block=1 #1-7
log_dir_path='results_0/logs'
model_dir_path='results_0/model'


# 2. compress
COMP_LOG=${log_dir_path}/compress_model_${block}.log
python compress_model.py \
    --block_id ${block} \
	--log_dir_path ${log_dir_path} \
	--model_dir_path ${model_dir_path} 2>&1 | tee ${COMP_LOG}