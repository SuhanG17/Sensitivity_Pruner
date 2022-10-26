#!/bin/bash
python Compress_Model.py --layer_id 9 --gpu_id 6,0,4 2>&1 | tee 'results/logs/compress_model_10.log'

