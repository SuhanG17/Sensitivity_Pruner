#!/bin/bash
gpus=2sh

for layer in $(seq 2 3)
do
	# 1. prune
	COMP_LOG=results/logs/compress_model_${layer}.log
	LOG=results/logs/log_${layer}.log
	python main.py \
		--gpu_id ${gpus} \
		--group_id ${layer} 2>&1 | tee ${LOG}

	# 3. compressed model
	python Compress_Model.py --group_id ${layer} 2>&1 | tee ${COMP_LOG}
done

python fine_tune_compressed_model.py --gpu_id ${gpus} 2>&1 | tee "results/logs/fine_tune_log.log"
