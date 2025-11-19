
export CUDA_VISIBLE_DEVICES=3

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=4
python retrieval_head_detection.py  --model_path /data/ylong030/huggingface/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07 --s 0 --e 2500



export CUDA_VISIBLE_DEVICES=2
python needle_in_haystack_with_mask.py \
    --s_len 0 \
    --e_len 50000 \
    --model_path /data/ylong030/huggingface/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07 \
    --model_provider LLaMA \
    --mask_topk 20



/data/ylong030/huggingface/hub/models--Dream-org--Dream-v0-Instruct-7B/snapshots/05334cb9faaf763692dcf9d8737c642be2b2a6ae
d149729398750b98c0af14eb82c78cfe92750796/

/data/ylong030/huggingface/hub/models--Dream-org--Dream-v0-Instruct-7B/snapshots/Dream
/data/ylong030/huggingface/hub/models--Qwen--Qwen2.5-7B/snapshots/qwen2.5

export CUDA_VISIBLE_DEVICES=2
python needle_in_haystack_with_mask.py \
    --s_len 0 \
    --e_len 50000 \
    --model_path /data/ylong030/huggingface/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07 \
    --model_provider LLaMA \
    --mask_topk 20


export CUDA_VISIBLE_DEVICES=2
python needle_in_haystack_with_mask.py \
    --s_len 0 \
    --e_len 2500 \
    --model_path /data/ylong030/huggingface/hub/models--Qwen--Qwen2.5-7B/snapshots/qwen2.5 \
    --model_provider LLaMA \
    --mask_topk 20



python retrieval_head_detection.py  --model_path /data/ylong030/huggingface/hub/models--Qwen--Qwen2.5-7B/snapshots/qwen2.5 --s 0 --e 2500


python retrieval_head_detection.py  --model_path /data/ylong030/huggingface/hub/models--Dream-org--Dream-v0-Instruct-7B/snapshots/dream --s 0 --e 2500



python retrieval_head_detection.py  --model_path /data/ylong030/huggingface/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07  --s 0 --e 2500

export CUDA_VISIBLE_DEVICES=2
python needle_in_haystack_with_mask.py \
    --s_len 0 \
    --e_len 2500 \
    --model_path /data/ylong030/huggingface/hub/models--Dream-org--Dream-v0-Instruct-7B/snapshots/dream\
    --model_provider LLaMA \
    --mask_topk 30
python needle_in_haystack_with_mask.py \
    --s_len 2500 \
    --e_len 5000 \
    --model_path /data/ylong030/huggingface/hub/models--Dream-org--Dream-v0-Instruct-7B/snapshots/dream\
    --model_provider LLaMA \
    --mask_topk 30
python needle_in_haystack_with_mask.py \
    --s_len 0 \
    --e_len 2500 \
    --model_path /data/ylong030/huggingface/hub/models--Dream-org--Dream-v0-Instruct-7B/snapshots/dream\
    --model_provider LLaMA \
    --mask_topk -30
python needle_in_haystack_with_mask.py \
    --s_len 2500 \
    --e_len 5000 \
    --model_path /data/ylong030/huggingface/hub/models--Dream-org--Dream-v0-Instruct-7B/snapshots/dream\
    --model_provider LLaMA \
    --mask_topk -30




conda activate retrieval_head
python needle_in_haystack_with_mask.py \
    --s_len 0 \
    --e_len 2500 \
    --model_path /data/ylong030/huggingface/hub/models--Qwen--Qwen2.5-7B/snapshots/qwen2.5 \
    --model_provider LLaMA \
    --mask_topk 30
python needle_in_haystack_with_mask.py \
    --s_len 2500 \
    --e_len 5000 \
    --model_path /data/ylong030/huggingface/hub/models--Qwen--Qwen2.5-7B/snapshots/qwen2.5 \
    --model_provider LLaMA \
    --mask_topk 30