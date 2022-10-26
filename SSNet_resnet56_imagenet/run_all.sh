#!/bin/bash
gpus=2,3,0sh
compression_rate=0.4

for layer in $(seq 0 3)
do
	# 1. prune
	COMP_LOG=results/logs/compress_model_${layer}.log
	LOG=results/logs/log_${layer}.log
	python main.py \
		--train_path 'data/imagenet_64_64/prune/train_loader' \
		--val_path 'data/imagenet_64_64/prune/val_loader' \
		--compression_rate ${compression_rate} \
		--gpu_id ${gpus} \
		--layersort \
		--group_id ${layer} 2>&1 | tee ${LOG}

	# 3. compressed model
	python Compress_Model.py --group_id ${layer} 2>&1 | tee ${COMP_LOG}
done

# python fine_tune_compressed_model.py --gpu_id ${gpus} 2>&1 | tee "results/logs/fine_tune_log.log"
