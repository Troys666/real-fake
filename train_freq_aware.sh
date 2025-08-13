#!/bin/bash

# 训练脚本 - 使用频率分解的Distribution Matching

export MODEL_NAME="/home/lkshpc/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14"

# 选择简单数据集 - 可以改成 imagewoof, imagefruit, imageyellow 等
SIMPLE_DATASET="imagenette"  # 只有10个类别

export OUTPUT_DIR="./LoRA/checkpoint/freq_aware_${SIMPLE_DATASET}"
export DATASET_NAME="/data/st/data/ILSVRC/Data/CLS-LOC/train" 
export LOG_DIR="./LoRA/logs"

echo "Training with frequency-aware distribution matching"
echo "Dataset: $SIMPLE_DATASET"
echo "Output directory: $OUTPUT_DIR"

# 创建输出目录
mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR

# 启动训练 - 使用频率感知的分布匹配损失
accelerate launch --mixed_precision="fp16" ./finetune/train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir $DATASET_NAME \
  --caption_column="text" \
  --dataset_subset=$SIMPLE_DATASET \
  --report_to=tensorboard \
  --resolution=512 --random_flip \
  --train_batch_size=3 \
  --num_train_epochs=10 \
  --checkpointing_steps=1000 \
  --learning_rate=1e-04 --lr_scheduler="constant" \
  --seed=42 \
  --output_dir=${OUTPUT_DIR} \
  --snr_gamma=5 \
  --guidance_token=8 \
  --dist_match=0.01 \
  --logging_dir $LOG_DIR \
  --max_train_samples=1000

echo "Training completed!"
echo "Model saved to: $OUTPUT_DIR"
