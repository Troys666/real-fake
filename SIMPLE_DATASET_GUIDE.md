# 简单数据集训练指南

## 可用的简单数据集

你的项目现在支持以下简单数据集子集：

### 推荐的简单数据集（按复杂度排序）

1. **imagenette** - 只有10个类别，最简单
   - 类别：如 tench, goldfish, great white shark 等
   - 数据量最小，训练最快

2. **imagewoof** - 10个狗品种
   - 类别：australian_terrier, border_terrier, samoyed, beagle 等
   - 适合测试动物图像生成

3. **imagefruit** - 10个水果类别
   - 类别：pineapple, banana, strawberry, orange 等
   - 适合测试食物图像生成

4. **imageyellow** - 10个黄色物体
   - 类别：bee, banana, lemon, corn, school_bus 等
   - 颜色一致性的有趣测试

5. **imagemeow** - 10个猫科动物
   - 类别：tabby_cat, bengal_cat, persian_cat, lion, tiger 等

## 使用方法

### 方法1：使用预配置的简单训练脚本

```bash
# 直接运行（默认使用imagenette数据集）
bash train_simple_dataset.sh
```

### 方法2：修改数据集类型

编辑 `train_simple_dataset.sh`，修改这一行：
```bash
SIMPLE_DATASET="imagenette"  # 改成其他数据集：imagewoof, imagefruit, imageyellow等
```

### 方法3：直接使用命令行参数

```bash
accelerate launch --mixed_precision="fp16" ./finetune/train_text_to_image_lora.py \
  --pretrained_model_name_or_path="/home/lkshpc/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14" \
  --train_data_dir="/data/st/data/ILSVRC/Data/CLS-LOC/train" \
  --dataset_subset="imagenette" \
  --caption_column="text" \
  --resolution=512 \
  --train_batch_size=8 \
  --num_train_epochs=10 \
  --learning_rate=1e-04 \
  --output_dir="./LoRA/checkpoint/simple_imagenette" \
  --max_train_samples=1000
```

## 训练参数说明

- `--dataset_subset`: 选择数据集子集
- `--max_train_samples`: 限制训练样本数量（可选）
- `--num_train_epochs`: 训练轮数（简单数据集建议10-20轮）
- `--train_batch_size`: 批次大小（可以设置更大，如8或16）

## 预期训练时间

- **imagenette (10类)**：约 10-30分钟（取决于硬件）
- **imagewoof (10类)**：约 10-30分钟
- **其他10类数据集**：类似时间

## 输出文件夹

训练完成后，模型权重会保存在：
```
./LoRA/checkpoint/simple_[数据集名称]/
```

## 验证训练结果

训练完成后可以用生成的LoRA权重进行图像生成测试，验证模型是否学会了对应类别的特征。

## 故障排除

如果遇到内存不足，可以：
1. 减小 `train_batch_size` 到 4 或更小
2. 减小 `resolution` 到 256
3. 设置 `max_train_samples` 限制样本数量
