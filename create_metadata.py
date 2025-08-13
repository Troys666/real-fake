#!/usr/bin/env python3
"""
为训练创建metadata.jsonl文件
这个脚本会扫描CLIP特征目录，为每个特征文件创建对应的训练条目
"""

import os
import json
from pathlib import Path

def create_metadata_jsonl():
    """创建metadata.jsonl文件"""
    
    # 设置路径
    features_dir = Path("/data/st/real-fake/LoRA/ImageNet1K_CLIPEmbedding/VIT_L")
    output_file = features_dir / "metadata.jsonl"
    
    print(f"扫描特征目录: {features_dir}")
    
    # 收集所有特征文件
    feature_files = []
    
    # 遍历所有类别目录
    for class_dir in features_dir.iterdir():
        if class_dir.is_dir() and class_dir.name.startswith('n'):
            print(f"处理类别: {class_dir.name}")
            
            # 遍历类别目录中的所有.pt文件
            for pt_file in class_dir.glob("*.pt"):
                # 提取原始文件名（去掉.pt扩展名）
                original_name = pt_file.stem
                
                # 构建相对路径
                relative_path = f"{class_dir.name}/{pt_file.name}"
                
                # 创建metadata条目
                metadata_entry = {
                    "file_name": f"{original_name}.jpg",  # 假设原始文件是jpg
                    "text": f"a photo of {class_dir.name}",  # 简单的描述文本
                    "clip_feature": relative_path  # CLIP特征文件的相对路径
                }
                
                feature_files.append(metadata_entry)
    
    print(f"找到 {len(feature_files)} 个特征文件")
    
    # 写入metadata.jsonl文件
    with open(output_file, 'w') as f:
        for entry in feature_files:
            json.dump(entry, f)
            f.write('\n')
    
    print(f"创建metadata.jsonl文件: {output_file}")
    print(f"总共 {len(feature_files)} 条训练数据")
    
    # 显示前几条示例
    print("\n前5条训练数据示例:")
    for i, entry in enumerate(feature_files[:5]):
        print(f"{i+1}. {entry}")

if __name__ == "__main__":
    create_metadata_jsonl()
