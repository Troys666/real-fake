#!/bin/bash

# 批量创建ImageNet类别文件夹的脚本
# 使用方法: bash create_folders.sh

SOURCE_DIR="/data/st/data/ILSVRC/Data/CLS-LOC/train"
OUTPUT_DIR="./LoRA/ImageNet1K_CLIPEmbedding/VIT_L"

echo "Creating output directory structure..."

# 创建输出根目录
mkdir -p "$OUTPUT_DIR"

# 获取所有n开头的类别文件夹并批量创建
echo "Copying directory structure from $SOURCE_DIR to $OUTPUT_DIR"

# 使用find和xargs进行高效批量创建
find "$SOURCE_DIR" -maxdepth 1 -type d -name "n*" -printf "%f\n" | \
xargs -I {} mkdir -p "$OUTPUT_DIR/{}"

echo "Directory structure creation completed!"
echo "Created $(find "$OUTPUT_DIR" -maxdepth 1 -type d -name "n*" | wc -l) class directories"
