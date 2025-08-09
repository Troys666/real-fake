import os
import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

def get_image_paths_from_file(file_path):
    """
    Extracts the list of image paths from a text file. 
    Each line in the file is assumed to have the format '/path/to/image.jpeg number'
    
    Args:
    file_path (str): The path to the text file.

    Returns:
    List[str]: A list of image paths.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    
    image_paths, labels = [], []
    for line in lines:
        # if if_imagenette:
        #     class_name = line.split()[0].split('/')[1]
        #     if not class_name in _LABEL_MAP:
        #         continue
        image_paths.append(line.split()[0])
        labels.append(line.split()[-1])
        
    # image_paths = [line.split()[0] for line in lines]
    # labels = [line.split()[-1] for line in lines]
    # print(image_paths)
    return image_paths, labels

def mirror_directory_structure(img_paths, source_directory, dest_directory):
    """
    Creates a mirror of the directory structure of source_directory in dest_directory.
    Uses efficient batch creation method to avoid performance issues.
    
    Args:
    img_paths (list): List of image paths with structure 'class_name/image_name.jpeg'.
    source_directory (str): The path of the source directory.
    dest_directory (str): The path of the destination directory.

    Returns:
    None
    """
    
    # 高效方式：直接从源目录获取所有类别文件夹并批量创建
    print(f"Creating directory structure in {dest_directory}...")
    
    # 确保目标根目录存在
    os.makedirs(dest_directory, exist_ok=True)
    
    # 获取源目录中的所有类别文件夹（n开头的）
    source_classes = [d for d in os.listdir(source_directory) 
                     if os.path.isdir(os.path.join(source_directory, d)) and d.startswith('n')]
    
    print(f"Found {len(source_classes)} class directories to create")
    
    # 批量创建目录，使用更高效的方式
    for i, class_name in enumerate(source_classes):
        dest_class_dir = os.path.join(dest_directory, class_name)
        if not os.path.exists(dest_class_dir):
            os.mkdir(dest_class_dir)  # 使用mkdir代替makedirs，因为父目录已存在
        
        # 每100个显示进度
        if (i + 1) % 100 == 0:
            print(f"Created {i + 1}/{len(source_classes)} directories")
    
    print("Directory structure creation completed!")
        
def create_ImageNetFolder(root_dir, out_dir):
    image_paths, labels = get_image_paths_from_file(os.path.join(root_dir,"file_list.txt")) 
    mirror_directory_structure(image_paths, root_dir, out_dir)

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    # 可以在这里添加测试代码
    pass