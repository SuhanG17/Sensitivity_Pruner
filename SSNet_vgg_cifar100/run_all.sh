#!/bin/bash
gpus=5,1,4sh 

for layer in $(seq 12 14)
do
	# 1. prune
	COMP_LOG=results/logs/compress_model_${layer}.txt
	LOG=results/logs/log_${layer}.txt
	python main.py \
		--gpu_id ${gpus} \
		--layer_id ${layer} 2>&1 | tee ${LOG}

	# 2. compressed model
	python Compress_Model.py --layer_id ${layer} 2>&1 | tee ${COMP_LOG}

done

python fine_tune_compressed_model.py --gpu_id ${gpus} 2>&1 | tee results/logs/log_fine_tune.txt
