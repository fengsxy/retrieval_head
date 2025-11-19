#!/bin/bash
# Set CUDA device
export CUDA_VISIBLE_DEVICES=2

# Define model paths
QWEN_PATH="/data/ylong030/huggingface/hub/models--Qwen--Qwen2.5-7B/snapshots/qwen2.5"
DREAM_PATH="/data/ylong030/huggingface/hub/models--Dream-org--Dream-v0-Instruct-7B/snapshots/dream"

## Run tests for Qwen
#python needle_in_haystack_with_mask.py \
#    --s_len 0 \
#    --e_len 2500 \
#    --model_path "$QWEN_PATH" \
#    --model_provider LLaMA \
#    --mask_topk 30
#
## Run tests for Dream (positive mask)
#python needle_in_haystack_with_mask.py \
#    --s_len 2500 \
#    --e_len 5000 \
#    --model_path "$DREAM_PATH" \
#    --model_provider LLaMA \
#    --mask_topk 30

# Run tests for Dream (negative mask)
python needle_in_haystack_with_mask.py \
    --s_len 0 \
    --e_len 2500 \
    --model_path "/data/ylong030/huggingface/hub/models--Dream-org--Dream-v0-Instruct-7B/snapshots/dream" \
    --model_provider LLaMA \
    --mask_topk -30

python needle_in_haystack_with_mask.py \
    --s_len 2500 \
    --e_len 5000 \
    --model_path "/data/ylong030/huggingface/hub/models--Dream-org--Dream-v0-Instruct-7B/snapshots/dream" \
    --model_provider LLaMA \
    --mask_topk -30
