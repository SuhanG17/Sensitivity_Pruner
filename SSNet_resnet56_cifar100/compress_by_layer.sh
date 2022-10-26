#!/bin/bash
gpus=2,3,7sh 

for layer in $(seq 3 3)
do
	# 1. prune
	COMP_LOG=results/logs/compress_model_${layer}.txt

	python Compress_Model.py --group_id ${layer} 2>&1 | tee ${COMP_LOG}

done

