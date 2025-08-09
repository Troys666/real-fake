#!/bin/bash
CUDA_VISIBLE_DEVICES=3 python extract_feature.py --index 0  --imagenet_path "/data/st/data/ILSVRC/Data/CLS-LOC/" #index分块处理数据 num_chunks=4

wait
echo "All processes completed"