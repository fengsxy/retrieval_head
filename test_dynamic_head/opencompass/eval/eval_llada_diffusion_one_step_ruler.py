"""RULER evaluation for LLaDA 8B using single-step diffusion decoding.

This mirrors ``eval_dynamic_llada_head_ruler.py`` but sets ``steps=1`` so the
entire completion is produced in a single denoising pass, which is useful for
testing the new long-range one-step diffusion pathway in ``llada_wrapper``.
"""

from __future__ import annotations

from pathlib import Path

from mmengine.config import read_base
from opencompass.partitioners import NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.models import LLaDACausalLM

with read_base():
    from opencompass.configs.datasets.ruler.ruler_4k_gen import ruler_datasets as ruler_datasets_4k
    from opencompass.configs.datasets.ruler.ruler_8k_gen import ruler_datasets as ruler_datasets_8k
    from opencompass.configs.datasets.ruler.ruler_16k_gen import ruler_datasets as ruler_datasets_16k
    from opencompass.configs.datasets.ruler.ruler_32k_gen import ruler_datasets as ruler_datasets_32k


datasets = []
datasets += ruler_datasets_4k
datasets += ruler_datasets_8k


# ----------------------------------------------------------------------------
# Common paths & configs
# ----------------------------------------------------------------------------
MODEL_PATH = "/data/ylong030/huggingface/hub/models--GSAI-ML--LLaDA-8B-Instruct/" \
    "snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07"
HEAD_SCORE_PATH = str((Path(__file__).resolve().parents[3] / "head_score/llada-block-2500.json").resolve())

COMMON_MODEL_KWARGS = dict(
    trust_remote_code=True,
    device_map="auto",
    torch_dtype="torch.bfloat16",
)
COMMON_DIFFUSION_CFG = dict(
    steps=1,
    block_length=32,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
)

BASE_SCALING_CFG = dict(
    scaling_factor=1,
    apply_dynamic_ntk_heads=False,
)



if not Path(HEAD_SCORE_PATH).exists():
    raise FileNotFoundError(f"Head score file not found: {HEAD_SCORE_PATH}")


models = [
    dict(
        type=LLaDACausalLM,
        abbr="llada_ntk32_baseline_onestep",
        path=MODEL_PATH,
        tokenizer_path=MODEL_PATH,
        model_kwargs=COMMON_MODEL_KWARGS,
        scaling_config=BASE_SCALING_CFG,
        diffusion_config=dict(COMMON_DIFFUSION_CFG),
        seed=2025,
        model_type="llada",
        max_seq_len=33000,
        max_out_len=32,
        batch_size=1,
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
]

work_dir = "./outputs/llada_one_step_ruler/"

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        task=dict(type=OpenICLInferTask),
    ),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=32,
        task=dict(type=OpenICLEvalTask, dump_details=True),
    ),
)

# Usage:
#   python run.py eval/eval_llada_diffusion_one_step_ruler.py --dump-eval-details -r
