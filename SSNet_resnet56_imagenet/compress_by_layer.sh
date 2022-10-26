#!/bin/bash
gpus=4,5sh 

# for layer in $(seq 3 3)
# do
# 	# 1. prune
# 	COMP_LOG=results/logs/compress_model_${layer}.txt

# 	python Compress_Model.py --group_id ${layer} 2>&1 | tee ${COMP_LOG}

# done

layer=0
# 1. prune
COMP_LOG=results/logs/compress_model_${layer}.txt

python Compress_Model.py --group_id ${layer} 2>&1 | tee ${COMP_LOG}


