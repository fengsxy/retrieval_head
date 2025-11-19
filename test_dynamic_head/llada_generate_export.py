# %% Cell 1
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")
print("CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])
print("Inside this notebook cuda:0 maps to the selected physical GPU.")

# %% Cell 2
from pathlib import Path
from typing import Dict, List, Optional
from pprint import pprint

import torch
from transformers import AutoTokenizer

from configuration_llada import LLaDAConfig
from modeling_llada import LLaDAModelLM

MODEL_PATH = Path("/data/ylong030/huggingface/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07").expanduser()
HEAD_SCORE_PATH = Path("../head_score/llada-block-2500.json").expanduser()
MASK_TOKEN_ID = 126336

DTYPE = torch.bfloat16
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

GEN_STEPS = 128
GEN_LENGTH = 128
BLOCK_LENGTH = 32
TEMPERATURE = 0.0
CFG_SCALE = 0.0
REMASKING = "low_confidence"  # or 'random'

ROPE_SCALING_FACTOR: Optional[float] = 4.0
HEAD_SCORE_TOP_K = 16

TEST_PROMPTS: List[str] = [
    "The capital of France is",
    "Explain how LLaDA's diffusion decode differs from autoregressive generation.",
]

print(f"Using device: {DEVICE}")
print(f"Model path: {MODEL_PATH}")
print(f"Head score path: {HEAD_SCORE_PATH}")

# %% Cell 3
config = LLaDAConfig.from_pretrained(str(MODEL_PATH))
config.use_cache = False  # diffusion decoding does not reuse KV cache

if ROPE_SCALING_FACTOR is not None:
    config.rope_scaling_factor = ROPE_SCALING_FACTOR

if HEAD_SCORE_PATH.exists():
    config.head_score_path = str(HEAD_SCORE_PATH)
    config.head_score_top_k = HEAD_SCORE_TOP_K
    config.head_score_threshold = None
else:
    print(f"⚠️  head score file not found: {HEAD_SCORE_PATH}")

tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = LLaDAModelLM.from_pretrained(
    str(MODEL_PATH),
    config=config,
    torch_dtype=DTYPE,
)
model.to(DEVICE)
model.eval()

print("Model dtype:", next(model.parameters()).dtype)
print("Configured rope scaling factor:", getattr(model.config, "rope_scaling_factor", "n/a"))
scaled_heads = getattr(model.config, "scaled_heads_dict", {}) or {}
print("Number of scaled head layers:", len(scaled_heads))
if scaled_heads:
    preview = dict(list(scaled_heads.items())[:3])
    print("Preview of per-layer scaled heads:")
    pprint(preview)

# %% Cell 4
import numpy as np
import torch.nn.functional as F

def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0:
        return logits
    logits64 = logits.to(torch.float64)
    noise = torch.rand_like(logits64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits64.exp() / gumbel_noise

def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    plan = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for row in range(mask_num.size(0)):
        extra = int(remainder[row].item())
        if extra > 0:
            plan[row, :extra] += 1
    return plan

# %% Cell 5
@torch.inference_mode()
def llada_decode(
    model: LLaDAModelLM,
    prompt_ids: torch.Tensor,
    steps: int = GEN_STEPS,
    gen_length: int = GEN_LENGTH,
    block_length: int = BLOCK_LENGTH,
    temperature: float = TEMPERATURE,
    cfg_scale: float = CFG_SCALE,
    remasking: str = REMASKING,
    mask_id: int = MASK_TOKEN_ID,
) -> torch.Tensor:
    device = next(model.parameters()).device
    prompt_ids = prompt_ids.to(device)
    batch_size, prompt_len = prompt_ids.shape
    total_len = prompt_len + gen_length
    x = torch.full((batch_size, total_len), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = prompt_ids
    prompt_mask = (x != mask_id)

    assert gen_length % block_length == 0, "gen_length must be divisible by block_length"
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0, "steps must be divisible by (gen_length / block_length)"
    steps_per_block = steps // num_blocks

    for block_idx in range(num_blocks):
        block_start = prompt_len + block_idx * block_length
        block_end = block_start + block_length
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for step_idx in range(steps_per_block):
            mask_index = (x == mask_id)
            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_mask] = mask_id
                x_in = torch.cat([x, un_x], dim=0)
                logits = model(input_ids=x_in, use_cache=False).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(input_ids=x, use_cache=False).logits

            logits_with_noise = add_gumbel_noise(logits, temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == "low_confidence":
                probs = F.softmax(logits, dim=-1)
                gather_index = x0.unsqueeze(-1)
                x0_p = torch.squeeze(torch.gather(probs, dim=-1, index=gather_index), -1)
            elif remasking == "random":
                x0_p = torch.rand((batch_size, total_len), device=device)
            else:
                raise NotImplementedError(f"Unknown remasking strategy: {remasking}")

            x0_p[:, block_end:] = float('-inf')
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, torch.full_like(x0_p, float('-inf')))

            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            for row in range(batch_size):
                quota = int(num_transfer_tokens[row, step_idx].item())
                if quota <= 0:
                    continue
                quota = min(quota, confidence.shape[1])
                _, idx = torch.topk(confidence[row], k=quota)
                transfer_index[row, idx] = True
            x[transfer_index] = x0[transfer_index]

    return x

# %% Cell 6
def decode_prompt(prompt: str, **kwargs) -> str:
    encoded = tokenizer(prompt, return_tensors="pt")
    output_ids = llada_decode(
        model,
        encoded["input_ids"],
        **kwargs,
    )
    completion_ids = output_ids[:, encoded["input_ids"].shape[1]:]
    return tokenizer.decode(completion_ids[0], skip_special_tokens=True).strip()

def run_batch(prompts: List[str], **kwargs) -> None:
    for idx, prompt in enumerate(prompts, 1):
        print(f"Prompt {idx}: {prompt}")
        completion = decode_prompt(prompt, **kwargs)
        print(completion if completion else "[empty]")
        print("-" * 72)

# %% Cell 7
run_batch(
    TEST_PROMPTS,
    steps=GEN_STEPS,
    gen_length=GEN_LENGTH,
    block_length=BLOCK_LENGTH,
    temperature=TEMPERATURE,
    cfg_scale=CFG_SCALE,
    remasking=REMASKING,
)

# %% Cell 8
# 自定义 prompt，可直接修改并重新运行本单元
custom_prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"
print(decode_prompt(custom_prompt, gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH))

# %% Cell 9

