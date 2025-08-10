import os
import json
from PIL import Image
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset, Features, Value, Image as ImageFeature
import pandas as pd

class ImageNetWithCaptions(Dataset):
    def __init__(self, image_root, caption_root, split="train", subset="imagenet1k"):
        """
        ImageNet数据集，集成BLIP2生成的captions
        
        Args:
            image_root: ImageNet图像根目录 (如 /data/st/data/ILSVRC/Data/CLS-LOC/train)
            caption_root: BLIP2 caption JSON文件根目录 (如 /data/st/real-fake/ImageNet_BLIP2_caption_json/ImageNet_BLIP2_caption_json)
            split: 数据集分割 ("train" 或 "val")
            subset: 数据集子集 (imagenette, imagewoof等)
        """
        self.image_root = image_root
        self.caption_root = caption_root
        self.subset = subset
        
        # 导入子集定义
        import sys
        sys.path.append('/data/st/real-fake')
        from data.new_load_data import imagenet_subclass_dict, imagenet_classes
        
        self.imagenet_subclass_dict = imagenet_subclass_dict
        self.imagenet_classes = imagenet_classes
        
        # 收集所有数据
        self.data = []
        self._load_data()
    
    def _load_data(self):
        """加载图像路径和对应的captions，根据subset过滤"""
        # 获取子集对应的类别索引
        if self.subset in self.imagenet_subclass_dict:
            target_classes = self.imagenet_subclass_dict[self.subset]
            print(f"Using subset {self.subset} with {len(target_classes)} classes")
        else:
            print(f"Unknown subset {self.subset}, using all classes")
            target_classes = list(range(1000))
        
        # 获取所有类别文件夹
        class_dirs = [d for d in os.listdir(self.image_root) 
                     if os.path.isdir(os.path.join(self.image_root, d)) and d.startswith('n')]
        
        # 创建类别文件夹名到类别索引的映射
        class_dir_to_idx = {}
        for idx, class_dir in enumerate(sorted(class_dirs)):
            class_dir_to_idx[class_dir] = idx
        
        print(f"Found {len(class_dirs)} ImageNet classes, filtering to subset...")
        
        for class_dir in class_dirs:
            # 检查该类别是否在目标子集中
            class_idx = class_dir_to_idx.get(class_dir)
            if class_idx is None or class_idx not in target_classes:
                continue
            
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
                
                # 构建CLIP嵌入文件的相对路径
                # 从 n01440764/n01440764_10026.JPEG 变成 n01440764/n01440764_10026.pt
                image_name_without_ext = image_file.replace('.JPEG', '')
                clip_embedding_path = f"{class_dir}/{image_name_without_ext}.pt"
                
                self.data.append({
                    'image': image_path,
                    'text': caption,
                    'class_id': class_dir,
                    'img_features': clip_embedding_path  # CLIP嵌入文件的相对路径
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

def create_imagenet_caption_dataset(image_root, caption_root, split="train", subset="imagenet1k"):
    """
    创建包含BLIP2 captions的ImageNet数据集
    
    Args:
        image_root: ImageNet图像根目录
        caption_root: BLIP2 caption JSON文件根目录 
        split: 数据集分割
        subset: 数据集子集 (imagenette, imagewoof, imagefruit, imageyellow, imagenet1k等)
    
    返回HuggingFace Dataset格式，使用内置的图像特性自动加载
    """
    dataset = ImageNetWithCaptions(image_root, caption_root, split, subset)
    
    # 使用HuggingFace Dataset的内置图像特性
    features = Features({
        'image': ImageFeature(),  # 自动处理图像加载和转换
        'text': Value('string'),
        'img_features': Value('string')
    })
    
    # 转换为HuggingFace Dataset
    data_dict = {
        'image': [item['image'] for item in dataset.data],
        'text': [item['text'] for item in dataset.data], 
        'img_features': [item['img_features'] for item in dataset.data]
    }
    
    print(f"Creating HuggingFace dataset with {subset} subset and automatic image loading...")
    hf_dataset = HFDataset.from_dict(data_dict, features=features)
    
    return hf_dataset

if __name__ == "__main__":
    # 测试数据加载
    image_root = "/data/st/data/ILSVRC/Data/CLS-LOC/train"
    caption_root = "/data/st/real-fake/ImageNet_BLIP2_caption_json/ImageNet_BLIP2_caption_json"
    
    dataset = create_imagenet_caption_dataset(image_root, caption_root)
    print(f"Created dataset with {len(dataset)} samples")
    print(f"Columns: {dataset.column_names}")
    print(f"Sample: {dataset[0]}")
