#!/bin/bash

dataset='imagenette'
versions=('v1')
loras=('gt_dm')
methods=('SDI2I_LoRA')
guidance_tokens=('Yes')
SDXLs=('No')
image_strengths=(0.75)

length=${#versions[@]}
echo "start Generation Loop"
for ((i=0; i<$length; i++)); do
    ver="${versions[$i]}"
    lora="/data/st/real-fake/LoRA/checkpoint/simple_imagenette/pytorch_lora_weights.safetensors"
    method="${methods[$i]}"
    guidance_token="${guidance_tokens[$i]}"
    SDXL="${SDXLs[$i]}"
    cw="${cws[$i]}"
    imst="${image_strengths[$i]}"
    echo "$ver LoRA: $lora Method $method"
    # Iterate from 0-7, cover all case for nchunks <= 8
    
    CUDA_VISIBLE_DEVICES=0 python generate.py --index 0 --method $method --version $ver --batch_size 24 \
    --use_caption "blip2" --dataset $dataset --lora_path $lora --if_SDXL $SDXL --use_guidance $guidance_token \
    --img_size 512 --cross_attention_scale 0.5 --image_strength $imst --nchunks 1 \
    --imagenet_path "/data/st/data/ILSVRC/Data/CLS-LOC" \
    --syn_path "/data/st/real-fake/synthetic_data" > results/gen0.out 2>&1 &
    
    wait
    echo "All processes completed"
done