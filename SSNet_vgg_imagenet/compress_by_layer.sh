#!/bin/bash
# gpus=1,2,3sh 
gpus=0,1

	# 1. prune
python Compress_Model.py --layer_id 2 2>&1 | tee "results/logs/compress_model_2.txt"


