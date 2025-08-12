#!/bin/bash

# 使用简单数据集的训练脚本
export MODEL_NAME="/home/lkshpc/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14"

# 选择简单数据集 - 可以改成 imagewoof, imagefruit, imageyellow 等
SIMPLE_DATASET="imagenette"  # 只有10个类别

export OUTPUT_DIR="./LoRA/checkpoint/small_0.06_MMD_${SIMPLE_DATASET}"
export DATASET_NAME="/data/st/data/ILSVRC/Data/CLS-LOC/train" 
export LOG_DIR="./LoRA/logs"

echo "Training with simple dataset: $SIMPLE_DATASET"
echo "Output directory: $OUTPUT_DIR"

# 创建输出目录
mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR

# 启动训练 - 减少了训练参数以便更快完成
accelerate launch --mixed_precision="fp16" ./finetune/train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir $DATASET_NAME \
  --caption_column="text" \
  --dataset_subset=$SIMPLE_DATASET \
  --report_to=tensorboard \
  --resolution=512 --random_flip \
  --train_batch_size=4 \
  --num_train_epochs=10 \
  --checkpointing_steps=1000 \
  --learning_rate=1e-04 --lr_scheduler="constant" \
  --seed=42 \
  --output_dir=${OUTPUT_DIR} \
  --snr_gamma=5 \
  --guidance_token=8 \
  --dist_match=0.03 \
  --logging_dir $LOG_DIR \
  --max_train_samples=1000

echo "Training completed!"
