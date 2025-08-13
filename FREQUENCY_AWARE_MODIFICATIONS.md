# Frequency-Aware Distribution Matching for LoRA Training

## 概述

我们已经成功修改了 `train_text_to_image_lora.py` 以支持频率感知的Distribution Matching损失。这种方法将噪声预测分解为低频和高频成分，分别优化结构一致性和细节质量。

## 主要修改

### 1. 频率分解函数 (`frequency_decomposition`)

位置: lines 466-518

```python
def frequency_decomposition(tensor, method='wavelet', wavelet='db3', J=2):
    """
    将张量分解为高频和低频部分
    - 支持小波变换 (pytorch_wavelets) 和 FFT 两种方法
    - 自动回退到FFT如果小波库不可用
    """
```

### 2. 频率感知的Distribution Matching损失

位置: lines 935-993

主要改进:
- **按照run2d.py的逻辑在latent space中进行频率分解**
- 从noise prediction恢复到predicted latent，然后分解频率成分
- 分别计算低频和高频的Distribution Matching损失
- 使用权重组合: 低频70%, 高频30%

```python
# 从noise prediction恢复到latent space (与run2d.py一致)
alphas_cumprod = noise_scheduler.alphas_cumprod.to(device=latents.device)
alpha_t = alphas_cumprod[unified_timesteps].view(-1, 1, 1, 1)
sigma_t = ((1 - alphas_cumprod[unified_timesteps]) ** 0.5).view(-1, 1, 1, 1)

if noise_scheduler.config.prediction_type == "epsilon":
    # z_0 = (z_t - sigma_t * eps) / alpha_t^0.5
    predicted_latent = (unified_noisy_latents - sigma_t * unified_model_pred) / (alpha_t ** 0.5)
    target_latent = (unified_noisy_latents - sigma_t * unified_target) / (alpha_t ** 0.5)

# 在latent space中进行频率分解 (与run2d.py一致)
pred_low, pred_high = frequency_decomposition(predicted_latent, method='fft')
target_low, target_high = frequency_decomposition(target_latent, method='fft')

# 分别计算低频和高频损失
low_freq_dist_loss = F.mse_loss(model_pred_low_ws, target_low_ws, reduction="mean")
high_freq_dist_loss = F.mse_loss(model_pred_high_ws, target_high_ws, reduction="mean")

# 组合损失
dist_loss = (low_freq_dist_loss * 0.7 + high_freq_dist_loss * 0.3) * args.dist_match
```

## 训练参数

使用 `train_freq_aware.sh` 脚本启动训练:

```bash
./train_freq_aware.sh
```

关键参数:
- `--dist_match`: 0.003 (Distribution Matching权重)
- `--guidance_token`: 8.0 (CLIP特征引导权重)
- `--rank`: 4 (LoRA秩)
- `--resolution`: 512
- `--train_batch_size`: 4

## 预期效果

1. **结构一致性**: 低频损失(权重0.7)确保生成图像的整体结构与目标分布匹配
2. **细节质量**: 高频损失(权重0.3)提升纹理和边缘细节的质量
3. **训练稳定性**: 频率分离有助于平衡不同尺度特征的学习

## 技术细节

### 频率分解方法

1. **小波变换** (优先选择，如果pytorch_wavelets可用):
   - 使用Daubechies-3小波
   - 2级分解 (J=2)
   - 完美重构保证

2. **FFT分解** (备用方案):
   - 低通滤波器截止频率: min(H,W)/4
   - 基于频域圆形掩码
   - 实时计算，无需额外依赖

### 损失计算流程

1. 统一时间步采样 (200-600)
2. 噪声预测和目标分解
3. 加权求和计算分布统计
4. 分频段MSE损失计算
5. 加权组合最终损失

## 测试验证

运行以下测试确保一切正常:

```bash
# 测试频率分解功能
python3 test_frequency_decomposition.py

# 测试训练脚本语法
python3 test_modified_training.py
```

## 与原始实现的对比

| 特性 | 原始实现 | 频率感知实现 |
|------|----------|-------------|
| 分解空间 | 无 | Latent Space (与run2d.py一致) |
| 分解方法 | 无 | 小波/FFT |
| 损失计算 | 全频段统一 | 分频段加权 |
| 结构控制 | 有限 | 增强(低频70%) |
| 细节控制 | 有限 | 增强(高频30%) |
| 计算开销 | 低 | 中等 |
| 与run2d.py兼容性 | 无 | 高度一致 |

## 重要技术细节

### 空间转换逻辑
1. **Noise Prediction → Latent Recovery**:
   - 使用diffusion公式恢复predicted latent
   - 支持epsilon和v_prediction两种模式
   - 与run2d.py的DDS方法保持数学一致性

2. **频率分解位置**:
   - **关键改进**: 在latent space而非noise space进行分解
   - 更符合图像处理的直觉理解
   - 与run2d.py的DWT-DDS方法逻辑一致

## 注意事项

1. 需要安装 `pytorch_wavelets` 以获得最佳性能
2. FFT方法作为备用，确保兼容性
3. 低频权重(0.7)可根据数据集调整
4. 内存使用略有增加，但在可接受范围内

## 下一步

1. 运行完整训练测试训练效果
2. 根据生成质量调整频率权重
3. 考虑添加更多频率分解选项
4. 评估不同数据集的适用性
