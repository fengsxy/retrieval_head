"""
Runtime Monkey-Patch Utility for LLaDA Mixed RoPE
===================================================

This utility allows you to add mixed RoPE support to an already-loaded LLaDA model
without modifying the source code.

Usage:
    from mixed_rope_patch import apply_mixed_rope_patch
    
    model = load_your_llada_model()
    scaled_heads_dict = {0: {2, 5}, 1: {3, 7}}  # layer -> head indices
    apply_mixed_rope_patch(model, scaling_factor=4.0, scaled_heads_dict=scaled_heads_dict)
"""

import torch
import torch.nn as nn
from typing import Dict, Set, Tuple, Optional
from torch import einsum
import logging

log = logging.getLogger(__name__)


def create_mixed_rope_forward(original_rope, scaling_factor, config):
    """
    Create a new forward method for RoPE that supports per-head scaling.
    
    Args:
        original_rope: The original RotaryEmbedding instance
        scaling_factor: RoPE theta scaling factor
        config: Model config
        
    Returns:
        New forward function
    """
    # Store both theta values
    rope_theta_base = config.rope_theta
    rope_theta_scaled = config.rope_theta * scaling_factor
    
    # Access the private cache
    cache = original_rope._RotaryEmbedding__cache
    
    def get_rotary_embedding(seq_len: int, device: torch.device, use_scaled: bool = False):
        """Get RoPE embeddings with optional scaling."""
        cache_key_sin = "rope_pos_sin_scaled" if use_scaled else "rope_pos_sin"
        cache_key_cos = "rope_pos_cos_scaled" if use_scaled else "rope_pos_cos"
        
        if (
            (pos_sin := cache.get(cache_key_sin)) is not None
            and (pos_cos := cache.get(cache_key_cos)) is not None
            and pos_sin.shape[-2] >= seq_len
            and pos_cos.shape[-2] >= seq_len
        ):
            if pos_sin.device != device:
                pos_sin = pos_sin.to(device)
                cache[cache_key_sin] = pos_sin
            if pos_cos.device != device:
                pos_cos = pos_cos.to(device)
                cache[cache_key_cos] = pos_cos
            return pos_sin[:, :, :seq_len, :], pos_cos[:, :, :seq_len, :]

        with torch.autocast(device.type, enabled=False):
            dim = config.d_model // config.n_heads
            rope_theta = rope_theta_scaled if use_scaled else rope_theta_base
            inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float) / dim))
            seq = torch.arange(seq_len, device=device, dtype=torch.float)
            freqs = einsum("i , j -> i j", seq, inv_freq)
            positions = torch.cat((freqs, freqs), dim=-1)
            pos_sin, pos_cos = positions.sin()[None, None, :, :], positions.cos()[None, None, :, :]
        
        cache[cache_key_sin] = pos_sin
        cache[cache_key_cos] = pos_cos
        return pos_sin, pos_cos
    
    def mixed_rope_forward(q: torch.Tensor, k: torch.Tensor, scaled_head_indices: Optional[Set[int]] = None):
        """
        Forward pass with mixed RoPE support.
        
        Args:
            q: Query tensor (B, nh, T, hs)
            k: Key tensor (B, nh, T, hs)
            scaled_head_indices: Set of head indices to apply scaled RoPE
        """
        if config.rope_full_precision:
            q_, k_ = q.float(), k.float()
        else:
            q_, k_ = q, k

        with torch.autocast(q.device.type, enabled=False):
            query_len, key_len = q_.shape[-2], k_.shape[-2]
            
            # No scaling case
            if scaled_head_indices is None or len(scaled_head_indices) == 0 or scaling_factor == 1.0:
                pos_sin, pos_cos = get_rotary_embedding(key_len, q_.device, use_scaled=False)
                pos_sin = pos_sin.type_as(q_)
                pos_cos = pos_cos.type_as(q_)
                q_ = original_rope.apply_rotary_pos_emb(
                    pos_sin[:, :, key_len - query_len : key_len, :],
                    pos_cos[:, :, key_len - query_len : key_len, :],
                    q_,
                )
                k_ = original_rope.apply_rotary_pos_emb(pos_sin, pos_cos, k_)
            else:
                # Mixed RoPE case
                num_heads = q_.shape[1]
                all_heads = set(range(num_heads))
                base_head_indices = all_heads - scaled_head_indices
                
                # Get embeddings
                pos_sin_base, pos_cos_base = get_rotary_embedding(key_len, q_.device, use_scaled=False)
                pos_sin_scaled, pos_cos_scaled = get_rotary_embedding(key_len, q_.device, use_scaled=True)
                
                pos_sin_base = pos_sin_base.type_as(q_)
                pos_cos_base = pos_cos_base.type_as(q_)
                pos_sin_scaled = pos_sin_scaled.type_as(q_)
                pos_cos_scaled = pos_cos_scaled.type_as(q_)
                
                # Apply to base heads
                if base_head_indices:
                    base_list = sorted(list(base_head_indices))
                    q_[:, base_list, :, :] = original_rope.apply_rotary_pos_emb(
                        pos_sin_base[:, :, key_len - query_len : key_len, :],
                        pos_cos_base[:, :, key_len - query_len : key_len, :],
                        q_[:, base_list, :, :]
                    )
                    k_[:, base_list, :, :] = original_rope.apply_rotary_pos_emb(
                        pos_sin_base, pos_cos_base, k_[:, base_list, :, :]
                    )
                
                # Apply to scaled heads
                if scaled_head_indices:
                    scaled_list = sorted(list(scaled_head_indices))
                    q_[:, scaled_list, :, :] = original_rope.apply_rotary_pos_emb(
                        pos_sin_scaled[:, :, key_len - query_len : key_len, :],
                        pos_cos_scaled[:, :, key_len - query_len : key_len, :],
                        q_[:, scaled_list, :, :]
                    )
                    k_[:, scaled_list, :, :] = original_rope.apply_rotary_pos_emb(
                        pos_sin_scaled, pos_cos_scaled, k_[:, scaled_list, :, :]
                    )
        
        return q_.type_as(q), k_.type_as(k)
    
    return mixed_rope_forward


def patch_block_attention(block, layer_idx, scaled_heads_dict):
    """
    Patch a single block's attention method to pass scaled_head_indices.
    
    Args:
        block: LLaDABlock instance
        layer_idx: Layer index
        scaled_heads_dict: Dict mapping layer_idx -> set of head indices
    """
    original_attention = block.attention
    
    # Get scaled heads for this layer
    scaled_heads = scaled_heads_dict.get(layer_idx, set())
    
    def patched_attention(
        q, k, v,
        attention_bias=None,
        layer_past=None,
        use_cache=False,
        block_mask=None,
        output_attentions=False,
    ):
        """Patched attention that passes scaled_head_indices to RoPE."""
        B, T, C = q.size()
        dtype = k.dtype
        
        # Handle head masking from block_mask
        masked_head_indices: Tuple[int, ...] = ()
        if block_mask:
            masked: Set[int] = set()
            for layer_idx_mask, head_idx_mask in block_mask:
                if layer_idx_mask == layer_idx:
                    try:
                        head_idx_int = int(head_idx_mask)
                    except (TypeError, ValueError):
                        continue
                    if 0 <= head_idx_int < block.config.n_heads:
                        masked.add(head_idx_int)
            if masked:
                masked_head_indices = tuple(sorted(masked))
        
        # Apply layer norm to q, k
        if block.q_norm is not None and block.k_norm is not None:
            q = block.q_norm(q).to(dtype=dtype)
            k = block.k_norm(k).to(dtype=dtype)
        
        # Reshape
        q = q.view(B, T, block.config.n_heads, C // block.config.n_heads).transpose(1, 2)
        k = k.view(B, T, block.config.effective_n_kv_heads, C // block.config.n_heads).transpose(1, 2)
        v = v.view(B, T, block.config.effective_n_kv_heads, C // block.config.n_heads).transpose(1, 2)
        
        # Apply head masking
        if masked_head_indices:
            for head_idx in masked_head_indices:
                q[:, head_idx, :, :] = 0
        
        # Handle past key values
        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)
        
        present = (k, v) if use_cache else None
        query_len, key_len = q.shape[-2], k.shape[-2]
        
        # Apply RoPE with scaled heads
        if block.config.rope:
            q, k = block.rotary_emb(q, k, scaled_head_indices=scaled_heads)
        
        # Cast attention bias
        if attention_bias is not None:
            attention_bias = block._cast_attn_bias(
                attention_bias[:, :, key_len - query_len : key_len, :key_len], dtype
            )
        
        # Scaled dot product attention
        att = block._scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_bias,
            dropout_p=0.0 if not block.training else block.config.attention_dropout,
            is_causal=False,
        )
        if isinstance(att, tuple):
            att, _ = att
        # Reshape and project (native impl never returns attn weights)
        att = att.transpose(1, 2).contiguous().view(B, T, C)
        return block.attn_out(att), present
    
    # Replace the attention method
    block.attention = patched_attention
    log.info(f"Patched attention for layer {layer_idx}, scaled heads: {scaled_heads}")


def apply_mixed_rope_patch(
    model,
    scaling_factor: float,
    scaled_heads_dict: Dict[int, Set[int]],
    verbose: bool = True
):
    """
    Apply mixed RoPE patch to an existing LLaDA model.
    
    Args:
        model: LLaDA model instance (HuggingFace format)
        scaling_factor: RoPE theta scaling factor for selected heads
        scaled_heads_dict: Dict mapping layer_idx -> set of head indices to scale
        verbose: Whether to print progress
        
    Returns:
        The patched model (modified in-place)
    
    Example:
        >>> model = AutoModelForCausalLM.from_pretrained("path/to/llada")
        >>> scaled_heads = {0: {2, 5, 8}, 1: {1, 3}}
        >>> apply_mixed_rope_patch(model, scaling_factor=4.0, scaled_heads_dict=scaled_heads)
    """
    if verbose:
        print("="*80)
        print("Applying Mixed RoPE Patch")
        print("="*80)
        print(f"Scaling factor: {scaling_factor}")
        print(f"Layers affected: {len(scaled_heads_dict)}")
        total_heads = sum(len(heads) for heads in scaled_heads_dict.values())
        print(f"Total heads scaled: {total_heads}")
        print()
    
    # Add config attributes
    model.config.rope_scaling_factor = scaling_factor
    model.config.scaled_heads_dict = scaled_heads_dict
    
    # Access model structure
    if hasattr(model, 'model'):
        llada_model = model.model
    else:
        llada_model = model
    
    # Get number of layers
    print(llada_model.config)
    num_layers = llada_model.config.n_layers
    
    # Patch each layer
    for layer_idx in range(num_layers):
        # Access block
        if hasattr(llada_model.transformer, 'blocks'):
            block = llada_model.transformer.blocks[layer_idx]
        elif hasattr(llada_model.transformer, 'block_groups'):
            group_idx = layer_idx // llada_model.config.block_group_size
            block_in_group = layer_idx % llada_model.config.block_group_size
            block = llada_model.transformer.block_groups[group_idx][block_in_group]
        else:
            raise ValueError("Unknown model architecture")
        
        # Patch RoPE forward method
        if hasattr(block, 'rotary_emb'):
            original_forward = block.rotary_emb.forward
            new_forward = create_mixed_rope_forward(
                block.rotary_emb, 
                scaling_factor, 
                llada_model.config
            )
            block.rotary_emb.forward = new_forward
            
            if verbose and layer_idx in scaled_heads_dict:
                print(f"Layer {layer_idx}: Patched RoPE for heads {sorted(scaled_heads_dict[layer_idx])}")
        
        # Patch attention method
        patch_block_attention(block, layer_idx, scaled_heads_dict)
    
    if verbose:
        print()
        print("="*80)
        print("Patching Complete!")
        print("="*80)
    
    return model


# Example usage
if __name__ == "__main__":
    print(__doc__)
    
    print("\nExample:")
    print("-" * 80)
    print("""
    from transformers import AutoModelForCausalLM
    from mixed_rope_patch import apply_mixed_rope_patch
    import json
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        "/path/to/llada",
        torch_dtype=torch.bfloat16,
        device_map='auto',
        trust_remote_code=True
    )
    
    # Load head scores
    with open("head_score/llada.json") as f:
        raw_scores = json.loads(f.readline())
    
    # Select top-10 heads
    scored_heads = []
    for key, values in raw_scores.items():
        layer_idx, head_idx = map(int, key.split("-"))
        score = float(np.mean(values))
        scored_heads.append(((layer_idx, head_idx), score))
    scored_heads.sort(key=lambda x: x[1], reverse=True)
    
    # Build scaled_heads_dict
    scaled_heads_dict = {}
    for (layer_idx, head_idx), _ in scored_heads[:10]:
        if layer_idx not in scaled_heads_dict:
            scaled_heads_dict[layer_idx] = set()
        scaled_heads_dict[layer_idx].add(head_idx)
    
    # Apply patch
    apply_mixed_rope_patch(model, scaling_factor=4.0, scaled_heads_dict=scaled_heads_dict)
    
    # Now use model as normal!
    """)
