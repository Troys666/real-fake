import os
import argparse
import json
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, Subset
from diffusers.utils import load_image
from Generation.data.ImageNet1K import create_ImageNetFolder
from data.new_load_data import get_generation_dataset
from transformers import AutoProcessor, CLIPModel
from diffusers.image_processor import VaeImageProcessor

class ImgFeatureExtractor():
    def __init__(self,args):
        self.device = "cuda"
        self.args = args
        self.model = "VIT_L"
                 
    def extract_feature(self):
        bsz = 16
        # 优化：使用扁平化文件结构，避免创建太多子文件夹
        output_dir = f"./LoRA/ImageNet1K_CLIPEmbedding/{self.model}"
        os.makedirs(output_dir, exist_ok=True)
        ImageNetPath = self.args.imagenet_path
        dataset = "imagenet1k"
        real_dst_train = get_generation_dataset(ImageNetPath, split="train",subset=dataset,filelist="file_list.txt")
        dataloader = self.get_subdataset_loader(real_dst_train, bsz, num_chunks=1)
        # Model
        if self.model in ["VIT_L"]:
            model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
            processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
            
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            targets, image_paths, image_names, class_names = batch
            bs = len(image_paths)
            # 简化：直接从image_path构建输出文件名
            out_paths = []
            for idx in range(bs):
                # 从完整路径构建输出文件名
                # 例如：/data/st/data/ILSVRC/Data/CLS-LOC/train/n01440764/n01440764_10026 -> n01440764_n01440764_10026.pt
                img_path = image_paths[idx]  # 这个路径已经不包含.JPEG后缀
                parts = img_path.split('/')
                class_name = parts[-2]  # n01440764
                file_name = parts[-1]   # n01440764_10026
                out_filename = f"{class_name}_{file_name}.pt"
                out_paths.append(os.path.join(output_dir, out_filename))
            
            if os.path.exists(out_paths[-1]):
                continue

            if self.model in ["VIT_L"]:
                images = [Image.open(image_paths[idx]+'.JPEG') for idx in range(bs)]
                inputs = processor(images=images, return_tensors="pt").to(self.device)
                image_features = model.get_image_features(**inputs).to(torch.float16)
            
            for idx in range(bs):
                torch.save(image_features[idx], out_paths[idx])
                
    def get_subdataset_loader(self, real_dst_train, bsz, num_chunks=4):
        # split Task
        # num_chunks = 8
        chunk_size = len(real_dst_train) // num_chunks
        chunk_index = self.args.index
        if chunk_index == num_chunks-1:
            subset_indices = range(chunk_index*chunk_size, len(real_dst_train))
        else:
            subset_indices = range(chunk_index*chunk_size, (chunk_index+1)*chunk_size)
        subset_dataset = Subset(real_dst_train, indices=subset_indices)
        dataloader = DataLoader(subset_dataset, batch_size=bsz, shuffle=False, num_workers=4)
        return dataloader


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index",default=0,type=int,help="split task")
    parser.add_argument("--imagenet_path",default="",type=str,help="path to imagenet")
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    extractor = ImgFeatureExtractor(args)
    extractor.extract_feature()




if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()