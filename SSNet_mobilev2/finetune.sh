#!/bin/bash
start_epoch=0
num_epochs=300
model_dir_path='results_0/model'
channel_dim_path='results_0/logs/block_7/channel_dim.pth'
checkpoint_path='results_0/checkpoint'

# 3. finetune
LOG=/nas/guosuhan/auto-prune/logs/imagenet/results_8/logs/finetune_64_64_epoch300.log
python finetune.py \
	--train_path 'data/imagenet_64_64/finetune/train_loader' \
	--val_path 'data/imagenet_64_64/finetune/val_loader' \
	--start_epoch ${start_epoch} \
	--num_epochs ${num_epochs} \
	--model_dir_path ${model_dir_path} \
	--channel_dim_path ${channel_dim_path} \
	--checkpoint_path ${checkpoint_path} 2>&1 | tee ${LOG}
	# --checkpoint_path ${checkpoint_path} \
	# --resume 2>&1 | tee ${LOG} 
	# --checkpoint_path ${checkpoint_path} 2>&1 | tee ${LOG}
	