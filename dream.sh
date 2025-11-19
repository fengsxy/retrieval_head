#!/bin/bash
#source /home/ylong030/miniconda3/bin/activate retrieval_head
export CUDA_VISIBLE_DEVICES=3
python needle_in_haystack_with_mask.py \
    --s_len 0 \
    --e_len 2500 \
    --model_path "/data/ylong030/huggingface/hub/models--Dream-org--Dream-v0-Instruct-7B/snapshots/dream" \
    --model_provider LLaMA \
    --mask_topk  30
python needle_in_haystack_with_mask.py \
    --s_len 2500 \
    --e_len 5000 \
    --model_path "/data/ylong030/huggingface/hub/models--Dream-org--Dream-v0-Instruct-7B/snapshots/dream" \
    --model_provider LLaMA \
    --mask_topk  30
python needle_in_haystack_with_mask.py \
    --s_len 0 \
    --e_len 2500 \
    --model_path "/data/ylong030/huggingface/hub/models--Dream-org--Dream-v0-Instruct-7B/snapshots/dream" \
    --model_provider LLaMA \
    --mask_topk  -30
python needle_in_haystack_with_mask.py \
    --s_len 2500 \
    --e_len 5000 \
    --model_path "/data/ylong030/huggingface/hub/models--Dream-org--Dream-v0-Instruct-7B/snapshots/dream" \
    --model_provider LLaMA \
    --mask_topk  -30
python needle_in_haystack_with_mask.py \
    --s_len 0 \
    --e_len 2500 \
    --model_path "/data/ylong030/huggingface/hub/models--Dream-org--Dream-v0-Instruct-7B/snapshots/dream" \
    --model_provider LLaMA \
    --mask_topk  0
python needle_in_haystack_with_mask.py \
    --s_len 2500 \
    --e_len 5000 \
    --model_path "/data/ylong030/huggingface/hub/models--Dream-org--Dream-v0-Instruct-7B/snapshots/dream" \
    --model_provider LLaMA \
    --mask_topk  0