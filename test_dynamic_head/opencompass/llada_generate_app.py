import argparse
import json
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from configuration_llada import LLaDAConfig
from modeling_llada import LLaDAModelLM

try:  # matplotlib is optional; fall back to text-only mode if missing
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    plt = None


DEFAULT_MODEL_PATH = Path(
    "/data/ylong030/huggingface/hub/models--GSAI-ML--LLaDA-8B-Instruct/"
    "snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07"
).expanduser()
DEFAULT_HEAD_SCORE = (Path(__file__).resolve().parent.parent / "head_score/llada-block-2500.json").resolve()
DEFAULT_MASK_TOKEN_ID = 126336
DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


@dataclass
class TraceRow:
    batch_index: int
    positions: List[int]
    token_ids: List[int]


@dataclass
class TraceEvent:
    iteration: int
    block_index: int
    step_in_block: int
    rows: List[TraceRow]


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


def llada_decode_with_trace(
    model: LLaDAModelLM,
    prompt_ids: torch.Tensor,
    steps: int,
    gen_length: int,
    block_length: int,
    temperature: float,
    cfg_scale: float,
    remasking: str,
    mask_id: int,
) -> Dict[str, Any]:
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

    fill_order = torch.full((batch_size, total_len), -1, dtype=torch.int32, device=device)
    trace_events: List[TraceEvent] = []
    iteration = 0

    for block_idx in range(num_blocks):
        block_start = prompt_len + block_idx * block_length
        block_end = block_start + block_length
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for step_idx in range(steps_per_block):
            iteration += 1
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

            x0_p[:, block_end:] = float("-inf")
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, torch.full_like(x0_p, float("-inf")))

            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            rows: List[TraceRow] = []
            for row in range(batch_size):
                quota = int(num_transfer_tokens[row, step_idx].item())
                if quota <= 0:
                    rows.append(TraceRow(batch_index=row, positions=[], token_ids=[]))
                    continue
                quota = min(quota, confidence.shape[1])
                _, idx = torch.topk(confidence[row], k=quota)
                transfer_index[row, idx] = True

            newly_filled = transfer_index & mask_index
            x[transfer_index] = x0[transfer_index]

            for row in range(batch_size):
                filled_positions = torch.nonzero(newly_filled[row], as_tuple=False).view(-1)
                if filled_positions.numel() > 0:
                    fill_order[row, filled_positions] = iteration
                    token_ids = x[row, filled_positions].detach().cpu().tolist()
                    rows.append(
                        TraceRow(
                            batch_index=row,
                            positions=filled_positions.detach().cpu().tolist(),
                            token_ids=token_ids,
                        )
                    )
                else:
                    rows.append(TraceRow(batch_index=row, positions=[], token_ids=[]))

            trace_events.append(
                TraceEvent(
                    iteration=iteration,
                    block_index=block_idx,
                    step_in_block=step_idx,
                    rows=rows,
                )
            )

    return {
        "output_ids": x,
        "trace_events": trace_events,
        "fill_order": fill_order.detach().cpu(),
        "prompt_length": prompt_len,
    }


def visualize_fill_timeline(
    fill_order: torch.Tensor,
    prompt_len: int,
    save_path: Path,
    title: str,
) -> Optional[Path]:
    if plt is None:
        print("matplotlib is not available; skipping heatmap export.")
        return None

    save_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = fill_order.numpy().astype(float)
    matrix[matrix < 0] = math.nan
    gen_matrix = matrix[:, prompt_len:]

    fig, ax = plt.subplots(figsize=(12, 2 + 0.2 * gen_matrix.shape[0]))
    cax = ax.imshow(gen_matrix, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("Generated token position")
    ax.set_ylabel("Batch index")
    fig.colorbar(cax, ax=ax, label="Decode iteration")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def log_trace(trace_events: List[TraceEvent], tokenizer: AutoTokenizer) -> None:
    print("\n=== Decode Timeline ===")
    for event in trace_events:
        header = f"[iter {event.iteration:03d}] block={event.block_index} step_in_block={event.step_in_block}"
        print(header)
        for row in event.rows:
            if not row.positions:
                print(f"  batch {row.batch_index}: no tokens transferred")
                continue
            tokens = tokenizer.convert_ids_to_tokens(row.token_ids)
            token_preview = ", ".join(tokens[:6])
            print(
                f"  batch {row.batch_index}: positions {row.positions[:6]} "
                f"-> {token_preview}"
                + (" ..." if len(row.positions) > 6 else "")
            )


def auto_device(preferred: Optional[str]) -> torch.device:
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def load_model_and_tokenizer(
    model_path: Path,
    head_score_path: Optional[Path],
    dtype: torch.dtype,
    rope_scaling_factor: Optional[float],
) -> Dict[str, Any]:
    config = LLaDAConfig.from_pretrained(str(model_path))
    config.use_cache = False
    if rope_scaling_factor is not None:
        config.rope_scaling_factor = rope_scaling_factor
    if head_score_path and head_score_path.exists():
        config.head_score_path = str(head_score_path)
        config.head_score_top_k = 16
        config.head_score_threshold = None
    elif head_score_path:
        print(f"⚠️  head score file not found: {head_score_path}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = LLaDAModelLM.from_pretrained(
        str(model_path),
        config=config,
        torch_dtype=dtype,
    )

    return {"tokenizer": tokenizer, "model": model}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize the intermediate decoding process of LLaDA."
    )
    parser.add_argument("--prompt", type=str, default="The capital of France is", help="Prompt to decode.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Path to the LLaDA model.")
    parser.add_argument(
        "--head-score-path",
        type=Path,
        default=DEFAULT_HEAD_SCORE,
        help="Path to the head score JSON for dynamic heads.",
    )
    parser.add_argument("--mask-token-id", type=int, default=DEFAULT_MASK_TOKEN_ID, help="Mask token id.")
    parser.add_argument("--steps", type=int, default=128, help="Number of diffusion steps.")
    parser.add_argument("--gen-length", type=int, default=128, help="Total generation length.")
    parser.add_argument("--block-length", type=int, default=32, help="Number of tokens per block.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for Gumbel noise.")
    parser.add_argument("--cfg-scale", type=float, default=0.0, help="Classifier free guidance scale.")
    parser.add_argument(
        "--remasking",
        type=str,
        choices=["low_confidence", "random"],
        default="low_confidence",
        help="Remasking strategy.",
    )
    parser.add_argument("--rope-scaling-factor", type=float, default=4.0, help="NTK rope scaling factor.")
    parser.add_argument(
        "--dtype",
        type=str,
        choices=list(DTYPE_MAP.keys()),
        default="bfloat16",
        help="Torch dtype for model weights.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device string. Defaults to cuda:0 if available else cpu.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./outputs/llada_app"),
        help="Directory for visualization artifacts.",
    )
    parser.add_argument(
        "--save-trace-json",
        action="store_true",
        help="Persist the trace timeline as JSON alongside the heatmap.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = auto_device(args.device)
    dtype = DTYPE_MAP[args.dtype]

    resources = load_model_and_tokenizer(
        model_path=args.model_path,
        head_score_path=args.head_score_path,
        dtype=dtype,
        rope_scaling_factor=args.rope_scaling_factor,
    )
    tokenizer: AutoTokenizer = resources["tokenizer"]
    model: LLaDAModelLM = resources["model"]
    model.to(device)
    model.eval()

    print(f"Using device: {device}")
    print(f"Model path: {args.model_path}")
    print(f"Head score path: {args.head_score_path}")

    encoded = tokenizer(args.prompt, return_tensors="pt")
    trace_result = llada_decode_with_trace(
        model=model,
        prompt_ids=encoded["input_ids"],
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length,
        temperature=args.temperature,
        cfg_scale=args.cfg_scale,
        remasking=args.remasking,
        mask_id=args.mask_token_id,
    )
    output_ids: torch.Tensor = trace_result["output_ids"]
    completion_ids = output_ids[:, encoded["input_ids"].shape[1]:]
    completion = tokenizer.decode(completion_ids[0], skip_special_tokens=True).strip()
    print("\n=== Final Completion ===")
    print(textwrap.fill(completion, width=100))

    log_trace(trace_result["trace_events"], tokenizer)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    heatmap_path = args.output_dir / "fill_timeline.png"
    saved_path = visualize_fill_timeline(
        fill_order=trace_result["fill_order"],
        prompt_len=trace_result["prompt_length"],
        save_path=heatmap_path,
        title="LLaDA diffusion fill order (per token)",
    )
    if saved_path:
        print(f"\nHeatmap saved to: {saved_path}")

    if args.save_trace_json:
        trace_json = args.output_dir / "trace_events.json"
        serializable = [
            {
                "iteration": event.iteration,
                "block_index": event.block_index,
                "step_in_block": event.step_in_block,
                "rows": [
                    {
                        "batch_index": row.batch_index,
                        "positions": row.positions,
                        "token_ids": row.token_ids,
                    }
                    for row in event.rows
                ],
            }
            for event in trace_result["trace_events"]
        ]
        with trace_json.open("w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        print(f"Trace JSON saved to: {trace_json}")


if __name__ == "__main__":
    main()
