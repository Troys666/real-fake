import os
import json
from PIL import Image
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset
import pandas as pd

class ImageNetWithCaptions(Dataset):
    def __init__(self, image_root, caption_root, split="train"):
        """
        ImageNet数据集，集成BLIP2生成的captions
        
        Args:
            image_root: ImageNet图像根目录 (如 /data/st/data/ILSVRC/Data/CLS-LOC/train)
            caption_root: BLIP2 caption JSON文件根目录 (如 /data/st/real-fake/ImageNet_BLIP2_caption_json/ImageNet_BLIP2_caption_json)
            split: 数据集分割 ("train" 或 "val")
        """
        self.image_root = image_root
        self.caption_root = caption_root
        
        # 收集所有数据
        self.data = []
        self._load_data()
    
    def _load_data(self):
        """加载图像路径和对应的captions"""
        # 获取所有类别文件夹
        class_dirs = [d for d in os.listdir(self.image_root) 
                     if os.path.isdir(os.path.join(self.image_root, d)) and d.startswith('n')]
        
        print(f"Found {len(class_dirs)} ImageNet classes")
        
        for class_dir in class_dirs:
            # 加载对应的caption JSON文件
            caption_file = os.path.join(self.caption_root, f"{class_dir}.json")
            
            if not os.path.exists(caption_file):
                print(f"Warning: Caption file not found for class {class_dir}")
                continue
            
            with open(caption_file, 'r') as f:
                caption_dict = json.load(f)
            
            # 获取该类别下的所有图像
            class_path = os.path.join(self.image_root, class_dir)
            image_files = [f for f in os.listdir(class_path) if f.endswith('.JPEG')]
            
            for image_file in image_files:
                image_path = os.path.join(class_path, image_file)
                
                # 获取对应的caption
                if image_file in caption_dict:
                    caption = caption_dict[image_file]
                else:
                    # 如果没有找到caption，使用默认的ImageNet类别名
                    caption = f"an image of {class_dir}"
                    print(f"Warning: No caption found for {image_file}, using default")
                
                self.data.append({
                    'image': image_path,
                    'text': caption,
                    'class_id': class_dir,
                    'img_features': None  # 预留给特征提取
                })
        
        print(f"Loaded {len(self.data)} image-caption pairs")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'image': item['image'],
            'text': item['text'],
            'img_features': item['img_features']
        }

def create_imagenet_caption_dataset(image_root, caption_root, split="train"):
    """
    创建包含BLIP2 captions的ImageNet数据集
    
    返回HuggingFace Dataset格式
    """
    dataset = ImageNetWithCaptions(image_root, caption_root, split)
    
    # 转换为pandas DataFrame
    data_dict = {
        'image': [item['image'] for item in dataset.data],
        'text': [item['text'] for item in dataset.data],
        'img_features': [item['img_features'] for item in dataset.data]
    }
    
    # 创建HuggingFace Dataset
    hf_dataset = HFDataset.from_dict(data_dict)
    
    return hf_dataset

if __name__ == "__main__":
    # 测试数据加载
    image_root = "/data/st/data/ILSVRC/Data/CLS-LOC/train"
    caption_root = "/data/st/real-fake/ImageNet_BLIP2_caption_json/ImageNet_BLIP2_caption_json"
    
    dataset = create_imagenet_caption_dataset(image_root, caption_root)
    print(f"Created dataset with {len(dataset)} samples")
    print(f"Columns: {dataset.column_names}")
    print(f"Sample: {dataset[0]}")
