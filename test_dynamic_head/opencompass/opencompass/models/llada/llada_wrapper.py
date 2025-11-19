from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.device import is_npu_available
from transformers import AutoTokenizer
try:
    from configuration_llada import LLaDAConfig
    from modeling_llada import LLaDAModelLM
except:
    from .configuration_llada import LLaDAConfig
    from .modeling_llada import LLaDAModelLM
from opencompass.models.huggingface_above_v4_33 import HuggingFaceBaseModel
from opencompass.registry import MODELS
from opencompass.utils.logging import get_logger
from opencompass.utils.prompt import PromptList
from opencompass.models.base import LMTemplateParser

ROOT_DIR = Path(__file__).resolve().parents[4]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

try:
    from mixed_rope_patch import apply_mixed_rope_patch
except ImportError:  # pragma: no cover
    apply_mixed_rope_patch = None


def _set_seed(seed: Optional[int]):
    if seed is None:
        return
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _as_torch_dtype(value):
    if value is None:
        return None
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        mapping = {
            "torch.float16": torch.float16,
            "torch.float32": torch.float32,
            "torch.float": torch.float32,
            "torch.bfloat16": torch.bfloat16,
            "torch.float64": torch.float64,
            "auto": None,
        }
        if value in mapping:
            return mapping[value]
        value = value.replace("torch.", "")
        return getattr(torch, value, None)
    return value


def _load_head_scores(path: Path) -> List[tuple]:
    raw = path.read_text().strip()
    if not raw:
        raise ValueError(f"Head score file is empty: {path}")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = json.loads(path.open().readline())
    scored = []
    for key, values in data.items():
        try:
            layer_idx, head_idx = map(int, key.split('-'))
        except ValueError:
            continue
        if isinstance(values, (list, tuple)):
            vals = [float(v) for v in values]
            if not vals:
                continue
            score = sum(vals) / len(vals)
        else:
            score = float(values)
        scored.append(((layer_idx, head_idx), float(score)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def _select_scaled_heads(
    scored: List[tuple],
    top_k: Optional[int],
    threshold: Optional[float],
) -> Dict[int, set]:
    result: Dict[int, set] = {}
    total = 0
    for (layer_idx, head_idx), score in scored:
        if threshold is not None and score < threshold:
            break
        if top_k is not None and total >= top_k:
            break
        result.setdefault(layer_idx, set()).add(head_idx)
        total += 1
    return result


def _add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0:
        return logits
    logits64 = logits.to(torch.float64)
    noise = torch.rand_like(logits64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits64.exp() / gumbel_noise


def _get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    plan = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for row in range(mask_num.size(0)):
        extra = int(remainder[row].item())
        if extra > 0:
            plan[row, :extra] += 1
    return plan


def _diffusion_generate_one_step(
    model: LLaDAModelLM,
    x: torch.Tensor,
    prompt_mask: torch.Tensor,
    *,
    temperature: float,
    cfg_scale: float,
    mask_id: int,
) -> torch.Tensor:
    mask_value = x.new_full((), mask_id, dtype=torch.long)
    mask_index = (x == mask_value)
    if cfg_scale > 0.0:
        un_x = x.clone()
        un_x[prompt_mask] = mask_value
        x_in = torch.cat([x, un_x], dim=0)
        logits = model(input_ids=x_in, use_cache=False).logits
        logits, un_logits = torch.chunk(logits, 2, dim=0)
        logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
    else:
        logits = model(input_ids=x, use_cache=False).logits
    logits_with_noise = _add_gumbel_noise(logits, temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)
    return torch.where(mask_index, x0, x)


@torch.inference_mode()
def diffusion_generate(
    model: LLaDAModelLM,
    input_ids: torch.Tensor,
    *,
    steps: int,
    gen_length: int,
    block_length: int,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    remasking: str = 'low_confidence',
    mask_id: int = 126336,
) -> torch.Tensor:
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    batch_size, prompt_len = input_ids.shape
    total_len = prompt_len + gen_length
    x = torch.full((batch_size, total_len), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = input_ids
    prompt_mask = (x != mask_id)

    if steps == 1:
        # Fill all masked positions in a single forward pass (one-step diffusion).
        return _diffusion_generate_one_step(
            model,
            x,
            prompt_mask,
            temperature=temperature,
            cfg_scale=cfg_scale,
            mask_id=mask_id,
        )

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    for block_idx in range(num_blocks):
        block_start = prompt_len + block_idx * block_length
        block_end = block_start + block_length
        block_mask = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = _get_num_transfer_tokens(block_mask, steps_per_block)

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

            logits_with_noise = _add_gumbel_noise(logits, temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == 'low_confidence':
                probs = F.softmax(logits, dim=-1)
                gather_index = x0.unsqueeze(-1)
                x0_p = torch.squeeze(torch.gather(probs, dim=-1, index=gather_index), -1)
            elif remasking == 'random':
                x0_p = torch.rand((batch_size, total_len), device=device)
            else:  # pragma: no cover
                raise NotImplementedError(f'Unknown remasking strategy: {remasking}')

            x0_p[:, block_end:] = float('-inf')
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, torch.full_like(x0_p, float('-inf')))

            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            for row in range(batch_size):
                quota = int(num_transfer_tokens[row, step_idx].item())
                if quota <= 0:
                    continue
                quota = min(quota, confidence.shape[1])
                _, indices = torch.topk(confidence[row], k=quota)
                transfer_index[row, indices] = True
            x[transfer_index] = x0[transfer_index]

    return x


@MODELS.register_module()
class LLaDACausalLM(HuggingFaceBaseModel):
    def __init__(
        self,
        path: str,
        tokenizer_path: Optional[str] = None,
        tokenizer_kwargs: Optional[dict] = None,
        model_kwargs: Optional[dict] = None,
        tokenizer_only: bool = False,
        generation_kwargs: Optional[dict] = None,
        max_seq_len: Optional[int] = None,
        stop_words: Optional[List[str]] = None,
        drop_middle: bool = False,
        scaling_config: Optional[dict] = None,
        diffusion_config: Optional[dict] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        _set_seed(seed)
        self.logger = get_logger()
        self.path = path
        self.template_parser = LMTemplateParser()
        self.max_seq_len = max_seq_len
        self.drop_middle = drop_middle
        self.scaling_config = scaling_config or {}
        self.diffusion_config = dict(
            steps=128,
            block_length=32,
            temperature=0.0,
            cfg_scale=0.0,
            remasking='low_confidence',
            mask_id=126336,
        )
        if diffusion_config:
            self.diffusion_config.update(diffusion_config)
        self.generation_kwargs = generation_kwargs or {}
        self.stop_words = stop_words or []
        tokenizer_kwargs = tokenizer_kwargs or {}
        self.tokenizer_only = tokenizer_only
        self.tokenizer = self._load_tokenizer(tokenizer_path or path, tokenizer_kwargs)
        if not tokenizer_only:
            self.model = self._load_model(path, model_kwargs or {})
        else:
            self.model = None

    def _load_tokenizer(self, tokenizer_path: str, tokenizer_kwargs: dict):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, **tokenizer_kwargs)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _prepare_dynamic_heads(self) -> Dict[int, set]:
        head_path = self.scaling_config.get('head_score_path')
        if not head_path:
            raise ValueError('apply_dynamic_ntk_heads=True requires head_score_path')
        path = Path(head_path)
        if not path.exists():
            raise FileNotFoundError(f'head score file not found: {path}')
        scored = _load_head_scores(path)
        top_k = self.scaling_config.get('head_score_top_k')
        threshold = self.scaling_config.get('head_score_threshold')
        return _select_scaled_heads(scored, top_k, threshold)

    def _load_model(self, path: str, kwargs: dict):
        DEFAULT_KWARGS = dict(device_map='auto', trust_remote_code=True)
        model_kwargs = {**DEFAULT_KWARGS, **kwargs}
        dtype = model_kwargs.get('torch_dtype', torch.bfloat16)
        model_kwargs['torch_dtype'] = _as_torch_dtype(dtype)
        if is_npu_available():
            model_kwargs['device_map'] = 'npu'

        config = LLaDAConfig.from_pretrained(path)
        config.flash_attention = True

        scaling_factor = self.scaling_config.get('scaling_factor', 1.0)
        apply_dynamic = self.scaling_config.get('apply_dynamic_ntk_heads', False)
        if scaling_factor != 1.0 and not apply_dynamic:
            config.rope_theta = config.rope_theta * scaling_factor

        model = LLaDAModelLM.from_pretrained(path, config=config, **model_kwargs)
        model.eval()
        if apply_dynamic:
            if apply_mixed_rope_patch is None:
                raise ImportError('mixed_rope_patch module not available')
            scaled_heads = self._prepare_dynamic_heads()
            apply_mixed_rope_patch(model, scaling_factor or 1.0, scaled_heads, verbose=False)
        return model

    def _prepare_inputs(self, inputs: List[str]) -> Dict[str, torch.Tensor]:
        if isinstance(inputs[0], PromptList):  # pragma: no cover
            raise NotImplementedError('PromptList is not supported in this lightweight wrapper.')
        messages = []
        for item in inputs:
            if isinstance(item, str):
                messages.append(item)
            else:
                combined = ''.join(seg['prompt'] for seg in item)
                messages.append(combined)
        if self.drop_middle:
            encoded = self.tokenizer(messages, padding=False, truncation=False)['input_ids']
            tensors = torch.tensor(encoded)
            if tensors.shape[-1] > self.max_seq_len:
                half = self.max_seq_len // 2
                tensors = torch.cat([tensors[:, :half], tensors[:, -half:]], dim=-1)
            tokens = {'input_ids': tensors}
        else:
            tokens = self.tokenizer.batch_encode_plus(
                messages,
                return_tensors='pt',
                padding=True,
                truncation=True,
                add_special_tokens=True,
                max_length=self.max_seq_len,
            )
        return tokens

    def generate(
        self,
        inputs: List[str],
        max_out_len: int,
        min_out_len: Optional[int] = None,
        stopping_criteria: Optional[List[str]] = None,
        **kwargs,
    ) -> List[str]:
        if self.model is None:
            raise RuntimeError('Tokenizer-only mode does not support generation')
        tokens = self._prepare_inputs(inputs)
        tokens = {k: v.to(self.model.device) for k, v in tokens.items()}

        stops = set((stopping_criteria or []) + self.stop_words)
        target_len = max_out_len if max_out_len is not None else self.diffusion_config.get('gen_length', 128)
        diffusion_cfg = dict(self.diffusion_config)
        diffusion_cfg['gen_length'] = target_len
        outputs = diffusion_generate(self.model, tokens['input_ids'], **diffusion_cfg)
        completions = outputs[:, tokens['input_ids'].shape[1]:]
        decoded = self.tokenizer.batch_decode(completions, skip_special_tokens=True)
        for stop in stops:
            decoded = [text.split(stop)[0] for text in decoded]
        return decoded

    def get_token_len(self, prompt: str, add_special_tokens: bool = True) -> int:
        tokens = self.tokenizer(prompt, add_special_tokens=add_special_tokens)
        return len(tokens['input_ids'])
