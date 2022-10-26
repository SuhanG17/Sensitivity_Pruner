#!/bin/bash
compression_rate=0.5
model_dir_path='results_0/model'
log_dir_path='results_0/logs'

for block in $(seq 1 7)
do
	# 1. prune
	LOG=${log_dir_path}/log_${block}.log
	python main.py \
		--train_path 'data/imagenet_64_64/prune/train_loader' \
		--val_path 'data/imagenet_64_64/prune/val_loader' \
		--model_dir_path ${model_dir_path} \
		--log_dir_path ${log_dir_path} \
		--compression_rate ${compression_rate} \
		--layersort \
		--block_id ${block} 2>&1 | tee ${LOG} 
	
	# 2. compress
	COMP_LOG=${log_dir_path}/compress_model_${block}.log
	python compress_model.py \
		--block_id ${block} \
		--log_dir_path ${log_dir_path} \
		--model_dir_path ${model_dir_path} 2>&1 | tee ${COMP_LOG}
done

# python fine_tune_compressed_model.py --gpu_id ${gpus} 2>&1 | tee "results/logs/fine_tune_log.log"
