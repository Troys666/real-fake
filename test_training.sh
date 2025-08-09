#!/bin/bash

# 测试训练脚本的数据加载部分
export MODEL_NAME="/home/lkshpc/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
export DATASET_NAME="/data/st/data/ILSVRC/Data/CLS-LOC/train"
export OUTPUT_DIR="./test_output"

echo "Testing training script with BLIP2 captions..."

python ./finetune/train_text_to_image_lora.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --train_data_dir $DATASET_NAME \
    --caption_column="text" \
    --resolution=512 \
    --train_batch_size=2 \
    --max_train_steps=5 \
    --learning_rate=1e-04 \
    --output_dir=${OUTPUT_DIR} \
    --checkpointing_steps=10 \
    --seed=42 \
    --dry_run
