"""
This script is adapted from 
https://github.com/gkamradt/LLMTest_NeedleInAHaystack

# GPT-4
(
python -u needle_in_haystack.py --s_len 0 --e_len 128000\
    --model_provider OpenAI\
    --model_name gpt-4-1106-preview
    --api_key $OPENAI_API_KEY
) 2>&1  | tee logs/eval_gpt_4_128k.log

# LLaMA 2 32K. Remember to download the model first
(
python -u needle_in_haystack.py --s_len 0 --e_len 128000\
    --model_provider LLaMA\
    --model_path ../../../Llama-2-7B-32K-Instruct
) 2>&1  | tee logs/eval_llama2_32k_instruct.log

# LongChat. Remember to download the model first
(
python -u needle_in_haystack.py --s_len 0 --e_len 128000\
    --model_provider LLaMA\
    --model_path /ML-A800/models/longchat-7b-v1.5-32k
) 2>&1  | tee logs/eval_longchat.log

# Our llama-2-7b-80k, requires 4*80G A100
# require you to download the model first
(
python -u needle_in_haystack.py --s_len 0 --e_len 128000\
    --model_provider LLaMA\
    --model_path ../../../llama-2-7b-80k
) 2>&1  | tee logs/eval_llama-2-7b-80k.log
"""

#import tiktoken
import os 
import glob
import json
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoConfig
import sys
sys.path.append("./faiss_attn/")
from source.modeling_llama import LlamaForCausalLM
from source.modeling_qwen2 import Qwen2ForCausalLM
from source.modeling_mixtral import MixtralForCausalLM
from source.modeling_mistral import MistralForCausalLM
#from source.modeling_phi3 import Phi3ForCausalLM
from source.modeling_llada import LLaDAModelLM
from source.configuration_llada import LLaDAConfig
from source.modeling_dream import DreamModel
from source.generation_utils import sample_tokens
import numpy as np
import argparse
from rouge_score import rouge_scorer
from datetime import datetime, timezone
from collections import defaultdict
import time
import torch
from lladault import add_gumbel_noise,get_num_transfer_tokens
import torch.nn.functional as F

def reset_rope(model, model_max_train_len, scaling_factor):
    for l in model.model.layers:
        l.self_attn.rotary_emb.scaling_factor = scaling_factor
        l.self_attn.rotary_emb._set_cos_sin_cache(seq_len=model_max_train_len, device=l.self_attn.rotary_emb.inv_freq.device, dtype=torch.float32)
    return
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

class LLMNeedleHaystackTester:
    """
    This class is used to test the LLM Needle Haystack.
    """
    def __init__(self,
                needle="\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n",
                haystack_dir="./haystack_for_detect",
                retrieval_question="What is the best thing to do in San Francisco?",
                results_version = 1,
                context_lengths_min = 1000,
                context_lengths_max = 50000,
                context_lengths_num_intervals = 20,
                context_lengths = None,
                document_depth_percent_min = 0,
                document_depth_percent_max = 100,
                document_depth_percent_intervals = 10,
                document_depth_percents = None,
                document_depth_percent_interval_type = "linear",
                model_provider = "OpenAI",
                model_name='',
                model_name_suffix=None,
                num_concurrent_requests = 1,
                save_results = True,
                save_contexts = True,
                final_context_length_buffer = 200,
                seconds_to_sleep_between_completions = None,
                print_ongoing_status = True):
        """        
        :param needle: The needle to be found in the haystack. Default is None.
        :param haystack_dir: The directory of text files to use as background context (or a haystack) in which the needle is to be found. Default is Paul Graham Essays.
        :param retrieval_question: The question which with to prompt the model to do the retrieval.
        :param results_version: In case you would like to try the same combination of model, context length, and depth % multiple times, change the results version other than 1
        :param num_concurrent_requests: Due to volume, this object is set up to run concurrent requests, default = 1. Be careful of rate limits.
        :param save_results: Whether or not you would like to save your contexts to file. Warning: These will get long! Default = True
        :param save_contexts: Whether or not you would like to save your contexts to file. Warning: These will get long! Default is True.
        :param final_context_length_buffer: The amount of cushion you'd like to leave off the input context to allow for the output context. Default 200 tokens
        :param context_lengths_min: The minimum length of the context. Default is 1000.
        :param context_lengths_max: The maximum length of the context. Default is 200000.
        :param context_lengths_num_intervals: The number of intervals for the context length. Default is 35.
        :param context_lengths: The lengths of the context. Default is None.
        :param document_depth_percent_min: The minimum depth percent of the document. Default is 0.
        :param document_depth_percent_max: The maximum depth percent of the document. Default is 100.
        :param document_depth_percent_intervals: The number of intervals for the document depth percent. Default is 35.
        :param document_depth_percents: The depth percentages of the document. Default is None.
        :param document_depth_percent_interval_type: The type of interval for the document depth percent. Must be either 'linear' or 'sigmoid'. Default is 'linear'.
        :param model_provider: The provider of the model. Must be either 'OpenAI' or 'Anthropic'. Default is 'OpenAI'.
        :param openai_api_key: The API key for OpenAI. Default is None.
        :param anthropic_api_key: The API key for Anthropic. Default is None.
        :param model_name: The name of the model. Default is 'gpt-4-1106-preview'.
        :param seconds_to_sleep_between_completions: The number of seconds to sleep between completions. Default is None.
        :param print_ongoing_status: Whether or not to print the ongoing status. Default is True.
        """
        if not needle or not haystack_dir or not retrieval_question:
            raise ValueError("Needle, haystack, and retrieval_question must be provided.")
        needles_and_stacks = [json.loads(l) for l in open(f"{haystack_dir}/needles.jsonl")]
        self.needle_list = [l["needle"] for l in needles_and_stacks]
        self.haystack_dir_list = [f"{haystack_dir}/part{i}" for i in range(1, 4)]
        self.retrieval_question_list = [l["question"] for l in needles_and_stacks]
        self.real_ansers_list = [l["real_needle"] for l in needles_and_stacks]
        self.results_version = results_version
        self.num_concurrent_requests = num_concurrent_requests
        self.save_results = save_results
        self.final_context_length_buffer = final_context_length_buffer
        self.save_contexts = save_contexts
        self.seconds_to_sleep_between_completions = seconds_to_sleep_between_completions
        self.print_ongoing_status = print_ongoing_status
        self.model_provider = model_provider
        self.testing_results = []
        self.head_counter = defaultdict(list)
        self.step_wise_head_counter = defaultdict(lambda: defaultdict(list))
        if("/" in model_name):
            self.model_version = model_name.split("/")[-1]
        else: self.model_version = model_name
        if(model_name_suffix is not None): self.model_version += "_" + model_name_suffix

        if context_lengths is None:
            if context_lengths_min is None or context_lengths_max is None or context_lengths_num_intervals is None:
                raise ValueError("Either context_lengths_min, context_lengths_max, context_lengths_intervals need to be filled out OR the context_lengths_list needs to be supplied.")
            else:
                self.context_lengths = np.round(np.linspace(context_lengths_min, context_lengths_max, num=context_lengths_num_intervals, endpoint=True)).astype(int)
        else:
            self.context_lengths = context_lengths

        if document_depth_percents is None:
            if document_depth_percent_min is None or document_depth_percent_max is None or document_depth_percent_intervals is None:
                raise ValueError("Either document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals need to be filled out OR the document_depth_percents needs to be supplied.")
            else:
                if document_depth_percent_interval_type == 'linear':
                    self.document_depth_percents = np.round(np.linspace(document_depth_percent_min, document_depth_percent_max, num=document_depth_percent_intervals, endpoint=True)).astype(int)
                elif document_depth_percent_interval_type == 'sigmoid':
                    self.document_depth_percents = [self.logistic(x) for x in np.linspace(document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals)]
        else:
            self.document_depth_percents = document_depth_percents

        if document_depth_percent_interval_type not in [None, "linear", "sigmoid"]:
            raise ValueError("document_depth_percent_interval_type must be either None, 'linear' or 'sigmoid'. If you'd like your own distribution give a list of ints in via document_depth_percent_intervals")
        
        self.model_name = model_name
        print(model_name)
        self.enc = AutoTokenizer.from_pretrained(model_name, use_fast=False,trust_remote_code=True)
        print("loading from %s" % model_name)
        config = AutoConfig.from_pretrained(model_name,trust_remote_code=True)
        self.supports_attn_mode = True
        self.supports_use_cache = True
        self.is_llada = False
        self.is_dream = False
        self.layer_num, self.head_num = config.num_hidden_layers, config.num_attention_heads
        print(f"layer number: {self.layer_num}, head number {self.head_num}")
        if getattr(config, "model_type", "").lower() == "llada":
            self.model_to_test = LLaDAModelLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                ).eval()
            print("model_version",self.model_version)
            self.supports_attn_mode = False
            self.supports_use_cache = False
            self.model_to_test.config.use_cache = False
            self.is_llada = True
        elif "dream" in self.model_version:
            self.model_to_test = DreamModel.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True,
                    attn_implementation="eager"
                ).eval()
            self.supports_attn_mode = False
            self.supports_use_cache = False
            self.model_to_test.config.use_cache = False
            self.model_to_test.config.output_attentions = True
            self.is_dream = True
        elif "qwen" in self.model_version:
            self.model_to_test = Qwen2ForCausalLM.from_pretrained(
                    model_name,torch_dtype="auto",device_map='auto',trust_remote_code=True,use_flash_attention_2="flash_attention_2"
                ).eval()
        elif "Mixtral" in self.model_version:
            self.model_to_test = MixtralForCausalLM.from_pretrained(
                    model_name,torch_dtype="auto",device_map='auto',use_flash_attention_2="flash_attention_2",trust_remote_code=True,
                ).eval()
        elif "Mistral" in self.model_version:
            self.model_to_test = MistralForCausalLM.from_pretrained(
                    model_name,torch_dtype="auto",device_map='auto',use_flash_attention_2="flash_attention_2",trust_remote_code=True,
                ).eval()
        elif "Phi3" in self.model_version:
            self.model_to_test = Phi3ForCausalLM.from_pretrained(
                    model_name,torch_dtype="auto",device_map='auto',use_flash_attention_2="flash_attention_2",trust_remote_code=True,
                ).eval()
        else:
            print("model_version",self.model_version)
            self.model_to_test = LlamaForCausalLM.from_pretrained(model_name,
                use_flash_attention_2="flash_attention_2", torch_dtype=torch.bfloat16,device_map='auto').eval()
            
        if 'llama-2-7b-80k' in self.model_version:
            scaling_factor = 10
            reset_rope(self.model_to_test, model_max_train_len=81920, scaling_factor=scaling_factor)
            
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            self.multi_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"])>1
        else:
            self.multi_gpus = True
            
        self.model_to_test_description = model_name
        self.evaluation_model = None
        self.debug='debug'
        model_name = model_name.split('/')[-1]

    def logistic(self, x, L=100, x0=50, k=.1):
        if x == 0:
            return 0
        if x == 100:
            return 100
        return np.round(L / (1 + np.exp(-k * (x - x0))), 3)
    
    def bound_evaluate_and_log(self, *args):
        self.evaluate_and_log(*args)

    def run_test(self, args):
        # Run through each iteration of context_lengths and depths
        tasks = []
        from tqdm import tqdm
        #self.context_lengths = [self.context_lengths[0],self.context_lengths[1]]
        for context_length in tqdm(self.context_lengths):
            if context_length < args.s_len or context_length > args.e_len: continue
            for depth_percent in self.document_depth_percents:
                task = self.bound_evaluate_and_log(context_length, depth_percent)

    def retrieval_calculate(
        self,
        attention_matrix,
        retrieval_score,
        inp=None,
        step_token=None,
        topk=1,
        row_idx=-1,
        block_mode=False,
        block_row_range=None,
    ):
        """
        Unified retrieval calculation for both token-wise and block-wise modes.

        Args:
            attention_matrix: list of tensors [num_layers][batch][num_heads][query_len, key_len].
            retrieval_score: nested list [layer][head][[sum_score, step_info]].
            inp: current token id (torch tensor with shape [1]) — used in token-level mode.
            step_token: readable token string (for logging).
            topk: number of top attention positions to consider in token-wise mode.
            row_idx: current token position in sequence.
            block_mode: if True, compute over a block of query tokens.
            block_row_range: tuple (row_start, row_end) for current block's query indices.
                            required when block_mode=True.

        Behavior:
            - token_mode: accumulate attention-mass ratio over the needle span for the current query token.
            - block_mode: compute total attention mass over needle range within current block only.
        """

        # === block-level mode ===
        if block_mode:
            assert block_row_range is not None, "block_row_range=(start,end) must be provided in block_mode"
            row_start, row_end = block_row_range

            for layer_idx in range(self.layer_num):
                for head_idx in range(self.head_num):
                    attn_block = attention_matrix[layer_idx][0][head_idx]
                    block_attn = attn_block[row_start:row_end, :]
                    needle_ratio = (
                        block_attn[:, self.needle_start:self.needle_end].sum(dim=1) /
                        (block_attn.sum(dim=1) + 1e-8)
                    ).mean().item()
                    if needle_ratio > 1:
                        print("error")
                        print(needle_ratio)
                    retrieval_score[layer_idx][head_idx][0] += needle_ratio
                    retrieval_score[layer_idx][head_idx][1] = f"block[{row_start}:{row_end}]"

            return  # skip token-level logic


        # === token-level mode (attention-mass alignment) ===
        for layer_idx in range(self.layer_num):
            for head_idx in range(self.head_num):
                row = attention_matrix[layer_idx][0][head_idx][row_idx]  # shape: [seq_len]
                denom = row.sum().item()
                if denom == 0:
                    continue
                needle_mass = row[self.needle_start:self.needle_end].sum().item()
                needle_ratio = float(needle_mass / (denom + 1e-8))
                if needle_ratio <= 0:
                    continue
                retrieval_score[layer_idx][head_idx][0] += needle_ratio
                retrieval_score[layer_idx][head_idx][1] = step_token if step_token is not None else f"row[{row_idx}]"


            
    def retrieval_head_accumulate(self, retrieval_score):
        for layer_idx in range(self.layer_num):
            for head_idx in range(self.head_num):
                self.head_counter[f"{layer_idx}-{head_idx}"].append(retrieval_score[layer_idx][head_idx][0])
    @torch.no_grad()
    def llada_decode(
        self,
        model,
        prompt,
        steps=128,
        gen_length=128,
        block_length=128,
        temperature=0.,
        cfg_scale=0.,
        remasking='low_confidence',
        mask_id=126336,
        return_stepwise_scores=False,
    ):
        '''
        Args:
            model: Mask predictor.
            prompt: A tensor of shape (1, L).
            steps: Sampling steps, less than or equal to gen_length.
            gen_length: Generated answer length.
            block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
            temperature: Categorical distribution sampling temperature.
            cfg_scale: Unsupervised classifier-free guidance scale.
            remasking: Remasking strategy. 'low_confidence' or 'random'.
            mask_id: The toke id of [MASK] is 126336.
            return_stepwise_scores: If True, also return a list with per-step head scores.
        '''
        x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
        x[:, :prompt.shape[1]] = prompt.clone()

        prompt_index = (x != mask_id)
        retrieval_score = [[[0, ''] for _ in range(model.config.num_attention_heads)] 
                   for _ in range(model.config.num_hidden_layers)]
        stepwise_scores = [] if return_stepwise_scores else None
        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length

        assert steps % num_blocks == 0
        steps = steps // num_blocks

        for num_block in range(num_blocks):
            block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
            for i in range(steps):
                mask_index = (x == mask_id)
                if cfg_scale > 0.:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    output = model(x_, output_attentions=True)
                    attentions = output.attentions
                    row_idx = prompt.shape[1] + num_block * block_length + block_length - 1
                    
                    logits  = output.logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    output = model(x, output_attentions=True)
                    attentions = output.attentions
                    row_idx = prompt.shape[1] + num_block * block_length + block_length - 1
                    logits = output.logits
            
                #print(output.attentions)
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
                token_id = x0[0, row_idx].item()
                step_token = self.enc.convert_ids_to_tokens(token_id)
                row_start = prompt.shape[1] + num_block * block_length
                row_end   = prompt.shape[1] + (num_block + 1) * block_length
                if return_stepwise_scores:
                    prev_scores = [
                        [retrieval_score[layer_idx][head_idx][0] for head_idx in range(self.head_num)]
                        for layer_idx in range(self.layer_num)
                    ]
                self.retrieval_calculate(
                    attentions,
                    retrieval_score,
                    torch.tensor([token_id]),
                    step_token,
                    row_idx=row_idx,
                    block_mode=True,
                    block_row_range=(row_start, row_end),
                )
                if return_stepwise_scores:
                    stepwise_scores.append([
                        [
                            retrieval_score[layer_idx][head_idx][0] - prev_scores[layer_idx][head_idx]
                            for head_idx in range(self.head_num)
                        ]
                        for layer_idx in range(self.layer_num)
                    ])
                if remasking == 'low_confidence':
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
                elif remasking == 'random':
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                    transfer_index[j, select_index] = True
                x[transfer_index] = x0[transfer_index]

        if return_stepwise_scores:
            return x, retrieval_score, stepwise_scores
        return x, retrieval_score

    @torch.no_grad()
    def dream_decode(
        self,
        model,
        prompt,
        steps=128,
        max_new_tokens=128,
        temperature=0.0,
        top_p=None,
        top_k=None,
        alg="origin",
        alg_temp=None,
        eps=1e-3,
        mask_id=None,
        return_stepwise_scores=False,
    ):
        """
        Diffusion-based decoding helper aligned with DreamGenerationMixin._sample.

        Args mirror the Dream sampler with minimal defaults needed for retrieval scoring.
        """
        if mask_id is None:
            mask_id = getattr(model.config, "mask_token_id", None)
        if mask_id is None:
            raise ValueError("mask_id must be provided when Dream config lacks mask_token_id.")

        device = prompt.device
        batch_size, prompt_len = prompt.shape
        total_len = prompt_len + max_new_tokens
        x = torch.full((batch_size, total_len), mask_id, dtype=torch.long, device=device)
        x[:, :prompt_len] = prompt

        retrieval_score = [
            [[0, ""] for _ in range(model.config.num_attention_heads)]
            for _ in range(model.config.num_hidden_layers)
        ]
        stepwise_scores = [] if return_stepwise_scores else None

        timesteps = torch.linspace(1, eps, steps + 1, device=device)
        attention_mask = None
        tok_idx = None
        row_range = (prompt_len, total_len)

        for step in range(steps):
            mask_index = (x == mask_id)
            if not mask_index.any():
                break

            outputs = model(
                input_ids=x,
                attention_mask=attention_mask,
                position_ids=tok_idx,
                output_attentions=True,
                return_dict=True,
            )
            logits = outputs.logits
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

            if return_stepwise_scores:
                prev_scores = [
                    [retrieval_score[layer_idx][head_idx][0] for head_idx in range(self.head_num)]
                    for layer_idx in range(self.layer_num)
                ]

            self.retrieval_calculate(
                outputs.attentions,
                retrieval_score,
                block_mode=True,
                block_row_range=row_range,
            )

            if return_stepwise_scores:
                stepwise_scores.append([
                    [
                        retrieval_score[layer_idx][head_idx][0] - prev_scores[layer_idx][head_idx]
                        for head_idx in range(self.head_num)
                    ]
                    for layer_idx in range(self.layer_num)
                ])

            mask_logits = logits[mask_index]
            if mask_logits.shape[0] == 0:
                continue

            t = timesteps[step]
            s = timesteps[step + 1]

            if alg == "origin":
                p_transfer = 1 - s / t if step < steps - 1 else 1
                if p_transfer <= 0:
                    continue
                x0 = torch.full_like(x[mask_index], mask_id, device=device)
                transfer_index = torch.rand_like(x0, dtype=torch.float32, device=device) < p_transfer
                if transfer_index.any():
                    _, sampled = sample_tokens(
                        mask_logits[transfer_index],
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                    )
                    x0[transfer_index] = sampled
                x[mask_index] = x0.clone()
            else:
                if alg == "maskgit_plus":
                    confidence, sampled = sample_tokens(
                        mask_logits,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                    )
                elif alg == "topk_margin":
                    confidence, sampled = sample_tokens(
                        mask_logits,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        margin_confidence=True,
                    )
                elif alg == "entropy":
                    confidence, sampled = sample_tokens(
                        mask_logits,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        neg_entropy=True,
                    )
                else:
                    raise RuntimeError(f"Unknown Dream sampling algorithm: {alg}")

                num_mask_token = mask_index.sum() / mask_index.shape[0]
                transfer_tokens = int(num_mask_token * (1 - s / t)) if step < steps - 1 else int(num_mask_token)
                full_confidence = torch.full_like(x, -torch.inf, dtype=logits.dtype, device=device)
                full_confidence[mask_index] = confidence
                if transfer_tokens > 0:
                    if alg_temp is None or alg_temp == 0:
                        _, transfer_index = torch.topk(full_confidence, transfer_tokens)
                    else:
                        full_confidence = full_confidence / alg_temp
                        full_confidence = F.softmax(full_confidence, dim=-1)
                        transfer_index = torch.multinomial(full_confidence, num_samples=transfer_tokens)
                    x_candidate = torch.full_like(x, mask_id, device=device)
                    x_candidate[mask_index] = sampled.clone()
                    row_indices = torch.arange(x.size(0), device=device).unsqueeze(1).expand_as(transfer_index)
                    x[row_indices, transfer_index] = x_candidate[row_indices, transfer_index]

        if return_stepwise_scores:
            return x, retrieval_score, stepwise_scores
        return x, retrieval_score


    def decode(self, q_outputs, inp, decode_len, block_list=None):
        output, retrieval_score = [], [[[0, ''] for _ in range(self.head_num)] for _ in range(self.layer_num)]
        past_kv = q_outputs.past_key_values
        for step_i in range(decode_len):
            inp = inp.view(1, 1)
            if self.supports_use_cache:
                decode_kwargs = dict(
                    input_ids=inp,
                    past_key_values=past_kv,
                    use_cache=True,
                    output_attentions=True,
                )
                if self.supports_attn_mode:
                    decode_kwargs["attn_mode"] = "torch"
                outputs = self.model_to_test(**decode_kwargs)
                past_kv = outputs.past_key_values
                inp = outputs.logits[0, -1].argmax()
                step_token = self.enc.convert_ids_to_tokens(inp.item())
                output.append(inp.item())
                self.retrieval_calculate(outputs.attentions, retrieval_score, inp, step_token)
                if step_token=='<0x0A>' or inp.item()==144: break
            else:
                decode_kwargs = dict(
                    input_ids=inp,
                    use_cache=False,
                    output_attentions=True,
                )
                if self.supports_attn_mode:
                    decode_kwargs["attn_mode"] = "torch"
                outputs = self.model_to_test(**decode_kwargs)
                inp = outputs.logits[0, -1].argmax()
                step_token = self.enc.convert_ids_to_tokens(inp.item())
                output.append(inp.item())
                self.retrieval_calculate(outputs.attentions, retrieval_score, inp, step_token)
                if step_token=='<0x0A>' or inp.item()==144: break

        return output, retrieval_score 

    def find_needle_idx(self, needle):
        needle_ids = self.enc(needle, add_special_tokens=False)["input_ids"]
        print( self.enc.decode(needle_ids, skip_special_tokens=False))
        span_len = len(needle_ids)
        for i in range(len(self.prompt_ids)):            
            token_span = self.prompt_ids[i : i + span_len]
            span_ids = set(token_span.tolist())
            overlap = float(len(span_ids.intersection(set(needle_ids)))) / len(set(needle_ids))
            if(overlap > 0.9):
                return i, i + span_len
        return -1, -1

    def evaluate_and_log(self, context_length, depth_percent):
        # Checks to see if you've already checked a length/percent/version.
        # This helps if the program stop running and you want to restart later
        # Go generate the required length context and place your needle statement in
        context = self.generate_context(context_length, depth_percent)
        question = f"Based on the content of the book, Question: {self.retrieval_question}\nAnswer:"
        '''
        if self.model_version=="Qwen1.5-14B-Chat":
            context = "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\n" + context input_context = "f{context}\nquestion<|im_end|>\n<|im_start|>assistant\n
            question += '<|im_end|>\n<|im_start|>assistant\n'
            input_ids = self.enc(input_context , return_tensors="pt")['input_ids']
        '''
        if self.model_version in ["Mistral-7B-Instruct-v0.2", "Qwen1.5-14B-Chat"]:
            prompt = [
            {"role": "user", "content": f"<book>{context}</book>\nBased on the content of the book, Question: {self.retrieval_question}\nAnswer:"},
            ]
            input_ids = self.enc.apply_chat_template(conversation=prompt, tokenize=True,  add_generation_prompt=True, return_tensors='pt')
        elif self.is_llada or self.is_dream:
            prompt = [{"role": "user", "content": f"<book>{context}</book>\nBased on the content of the book, Question: {self.retrieval_question}\nAnswer:"}]
            print(prompt)
            print(self.real_needle)
            input_ids = self.enc.apply_chat_template(conversation=prompt, tokenize=True, add_generation_prompt=True, return_tensors='pt')
        else:
            input_context = context + question
            input_ids = self.enc(input_context , return_tensors="pt")['input_ids']

        # Prepare your message to send to the model you're going to evaluate
        test_start_time = time.time()
        self.prompt_ids = input_ids[0, :]
        if not self.multi_gpus:
            input_ids = input_ids.to(self.model_to_test.device)
        self.needle_start, self.needle_end = self.find_needle_idx(self.real_needle)
        stepwise_scores = None
        with torch.no_grad():
            if self.is_dream:
                output, retrieval_score, stepwise_scores = self.dream_decode(
                    self.model_to_test,
                    input_ids,
                    steps=32,
                    max_new_tokens=32,
                    temperature=0.0,
                    return_stepwise_scores=True,
                )
                output = output[:, input_ids.shape[1]:]
                response = self.enc.batch_decode(output, skip_special_tokens=True)[0]
            elif self.supports_use_cache:
                q_outputs = self.model_to_test(input_ids=input_ids[:,:-1], use_cache=True, return_dict=True)
                output, retrieval_score  = self.decode(q_outputs, input_ids[:,-1], 50)
                response = self.enc.decode(output,skip_special_tokens=True).strip()
            else:
                output, retrieval_score, stepwise_scores = self.llada_decode(
                    self.model_to_test,
                    input_ids,
                    steps=32,              # 等价于 decode_len
                    gen_length=32,         # 可设为相同
                    block_length=32,        # 可调
                    temperature=0.0,
                    cfg_scale=0.0,
                    remasking='low_confidence',
                    return_stepwise_scores=True,
                )
                #out = generate(model, input_ids, steps=128, gen_length=128, block_length=1, temperature=0., cfg_scale=0., remasking='low_confidence')
                output =output[:, input_ids.shape[1]:]
                response = self.enc.batch_decode(output,skip_special_tokens=True)[0]
                #print(tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0])
           #generate(model, input_ids, steps=128, gen_length=128, block_length=1, temperature=0., cfg_scale=0., remasking='low_confidence')
            
            print(response)
        test_end_time = time.time()
        test_elapsed_time = test_end_time - test_start_time
        
        score = scorer.score(self.real_needle, response)['rouge1'].recall*100
        context_file_location = f'{self.model_version.replace(".", "_")}_len_{context_length}_depth_{int(depth_percent*100)}'
        ## if recall > 50, we determine this retrieval succeed and update the retrieval score
        if score > 50:
            self.retrieval_head_accumulate(retrieval_score)
            head_score = [(i[0], np.mean(i[1])) for i in self.head_counter.items()]
            head_score = sorted(head_score, key=lambda x:x[1], reverse=True)
            if stepwise_scores:
                accumulated_scores = defaultdict(float)
                per_step_history = defaultdict(list)
                for step_idx, step_scores in enumerate(stepwise_scores, start=1):
                    step_key = f"step_{step_idx}"
                    for layer_idx in range(self.layer_num):
                        for head_idx in range(self.head_num):
                            head_key = f"{layer_idx}-{head_idx}"
                            increment = float(step_scores[layer_idx][head_idx])
                            accumulated_scores[head_key] += increment
                            per_step_history[head_key].append(accumulated_scores[head_key])
                            self.step_wise_head_counter[step_key][head_key].append(increment)
                top_heads = sorted(
                    accumulated_scores.items(), key=lambda x: x[1], reverse=True
                )[:5]
                if top_heads:
                    preview = {head: per_step_history[head] for head, _ in top_heads}
                    print(f"Stepwise accumulated head preview (top5): {preview}")
        else:
            print("retrieval failed")

        results = {
            'model' : self.model_to_test_description,
            'context_length' : int(context_length),
            'depth_percent' : float(depth_percent),
            'version' : self.results_version,
            'needle' : self.needle,
            'model_response' : response,
            'score' : score,
            'test_duration_seconds' : test_elapsed_time,
            'test_timestamp_utc' : datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S%z')
        }

        self.testing_results.append(results)

        if self.print_ongoing_status:
            print (f"-- Test Summary -- ")
            print (f"Duration: {test_elapsed_time:.1f} seconds")
            print (f"Context: {context_length} tokens")
            print (f"Depth: {depth_percent}%")
            print (f"Score: {score}")
            print (f"Response: {response}\n")

        if self.save_contexts:
            results['file_name'] : context_file_location

            # Save the context to file for retesting
            if not os.path.exists('contexts'):
                os.makedirs('contexts')

            if not os.path.exists(f'contexts/{self.model_version}'):
                os.makedirs(f'contexts/{self.model_version}')

            with open(f'contexts/{self.model_version}/{context_file_location}_context.txt', 'w') as f:
                f.write(context)
            
        if self.save_results:
            # Save the context to file for retesting
            if not os.path.exists(f'results/graph/{self.model_version}'):
                os.makedirs(f'results/graph/{self.model_version}')
            
            # Save the result to file for retesting
            p = f'results/graph/{self.model_version}/{context_file_location}_results.json'
            print("Writing at %s" % p)
            with open(p, 'w') as f:
                json.dump(results, f)

    def result_exists(self, context_length, depth_percent):
        """
        Checks to see if a result has already been evaluated or not
        """

        results_dir = 'results/' + self.model_version
        print("Searching existing results at %s" % results_dir)
        if not os.path.exists(results_dir):
            return False
        for filename in os.listdir(results_dir):
            if filename.endswith('.json'):
                with open(os.path.join(results_dir, filename), 'r') as f:
                    result = json.load(f)
                    context_length_met = result['context_length'] == context_length
                    depth_percent_met = result['depth_percent'] == depth_percent
                    version_met = result.get('version', 1) == self.results_version
                    model_met = result['model'] == self.model_name
                    # import ipdb; ipdb.set_trace()
                    if context_length_met and depth_percent_met and version_met and model_met:
                        return True
        return False

    def generate_context(self, context_length, depth_percent):
        # Load up tiktoken so we navigate tokens more easily

        # Get your Paul Graham files loaded into a string
        context = self.read_context_files()

        # Truncate the Paul Graham essays to the context length you desire
        context = self.encode_and_trim(context, context_length)

        # Insert your random statement according to your depth percent
        context = self.insert_needle(context, depth_percent, context_length)

        return context
    
    def encode_text_to_tokens(self, text):
        if self.model_provider in ["OpenAI", "LLaMA", "Mistral", "GLM"]:
            return self.enc.encode(text)
        elif self.model_provider == "Anthropic":
            # Assuming you have a different encoder for Anthropic
            return self.enc.encode(text).ids
        else:
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")
    
    def insert_needle(self, context, depth_percent, context_length):
        tokens_needle = self.encode_text_to_tokens(self.needle)
        tokens_context = self.encode_text_to_tokens(context)

        # Reducing the context length by 150 buffer. This is to account for system message, the user question, and response.
        context_length -= self.final_context_length_buffer

        # If your context + needle are longer than the context length (which it will be), then reduce tokens from the context by the needle length
        if len(tokens_context) + len(tokens_needle) > context_length:
            tokens_context = tokens_context[:context_length - len(tokens_needle)]

        if depth_percent == 100:
            # If your depth percent is 100 (which means your needle is the last thing in the doc), throw it at the end
            tokens_new_context = tokens_context + tokens_needle
        else:
            # Go get the position (in terms of tokens) to insert your needle
            insertion_point = int(len(tokens_context) * (depth_percent / 100))
            # import ipdb; ipdb.set_trace()

            # tokens_new_context represents the tokens before the needle
            tokens_new_context = tokens_context[:insertion_point]

            # We want to make sure that we place our needle at a sentence break so we first see what token a '.' is
            if(self.model_provider in ["LLaMA", "LongLLaMA"]): period_tokens = [29889, 869]
            elif(self.model_provider == "Mistral"): period_tokens = [842, 28723]
            elif(self.model_provider == "GLM"): period_tokens = [918, 30930]
            else: period_tokens = self.encode_text_to_tokens('.')
            
            # Then we iteration backwards until we find the first period
            while tokens_new_context and tokens_new_context[-1] not in period_tokens:
                insertion_point -= 1
                tokens_new_context = tokens_context[:insertion_point]

            print("insertion at %d" % insertion_point)
            # Once we get there, then add in your needle, and stick the rest of your context in on the other end.
            # Now we have a needle in a haystack
            tokens_new_context += tokens_needle + tokens_context[insertion_point:]

        # Convert back to a string and return it
        new_context = self.decode_tokens(tokens_new_context)
        return new_context

    def get_context_length_in_tokens(self, context):
        if self.model_provider in ["OpenAI", "LLaMA", "Mistral", "GLM"]:
            return len(self.enc.encode(context))
        elif self.model_provider == "Anthropic":
            # Assuming you have a different encoder for Anthropic
            encoded = self.enc.encode(context)
            return len(self.enc.encode(context).ids)
        else:
            
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")

    def read_context_files(self):
        context = ""
        max_context_length = max(self.context_lengths)

        while len(context.split()) < max_context_length:
            for file in glob.glob(f"{self.haystack_dir}/*.txt"):
                with open(file, 'r') as f:
                    context += f.read()
        return context

    def get_tokens_from_context(self, context):
        if self.model_provider in ["OpenAI", "LLaMA", "Mistral", "GLM"]:
            return self.enc.encode(context)
        elif self.model_provider == "Anthropic":
            # Assuming you have a different encoder for Anthropic
            return self.enc.encode(context).ids
        else:
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")
        
    def decode_tokens(self, tokens, context_length=None):
        if self.model_provider in ["OpenAI", "LLaMA", "Mistral", "GLM"]:
            return self.enc.decode(tokens[:context_length])
        elif self.model_provider == "Anthropic":
            # Assuming you have a different decoder for Anthropic
            return self.enc.decode(tokens[:context_length])
        else:
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")

    def encode_and_trim(self, context, context_length):
        tokens = self.get_tokens_from_context(context)
        if len(tokens) > context_length:
            context = self.decode_tokens(tokens, context_length)
        return context
    
    def get_results(self):
        return self.testing_results
    
    def print_start_test_summary(self):
        print ("\n")
        print ("Starting Needle In A Haystack Testing...")
        print (f"- Model: {self.model_name}")
        print (f"- Context Lengths: {len(self.context_lengths)}, Min: {min(self.context_lengths)}, Max: {max(self.context_lengths)}")
        print (f"- Document Depths: {len(self.document_depth_percents)}, Min: {min(self.document_depth_percents)}%, Max: {max(self.document_depth_percents)}%")
        print (f"- Needle: {self.needle.strip()}")
        print ("\n\n")

    def start_test(self, args):
        for ni in range(len(self.needle_list)):
            self.needle = self.needle_list[ni]
            self.haystack_dir = self.haystack_dir_list[ni]
            self.real_needle  = self.real_ansers_list[ni]
            self.retrieval_question = self.retrieval_question_list[ni]
            if self.print_ongoing_status:
                self.print_start_test_summary()
            self.run_test(args)
        os.makedirs("head_score", exist_ok=True)
        head_counter_path = f"head_score/{self.model_version}.json"
        if os.path.exists(head_counter_path):
            with open(head_counter_path, "r") as file:
                try:
                    head_counter = json.load(file)
                except json.JSONDecodeError:
                    head_counter = {}
            for k, v in head_counter.items():
                self.head_counter[k] += v
        with open(head_counter_path, 'w') as f:
            json.dump(self.head_counter, f)

        stepwise_path = f"head_score/{self.model_version}_stepwise.json"
        if os.path.exists(stepwise_path):
            with open(stepwise_path, "r") as file:
                try:
                    existing_stepwise = json.load(file)
                except json.JSONDecodeError:
                    existing_stepwise = {}
            if isinstance(existing_stepwise, dict):
                for step_key, head_scores in existing_stepwise.items():
                    if not isinstance(head_scores, dict):
                        continue
                    for head, values in head_scores.items():
                        if isinstance(values, list):
                            self.step_wise_head_counter[step_key][head].extend(values)

        serialized_stepwise = {
            step_key: {head: values for head, values in head_scores.items()}
            for step_key, head_scores in self.step_wise_head_counter.items()
            if head_scores
        }
        with open(stepwise_path, "w") as f:
            json.dump(serialized_stepwise, f)
        print(f"Saved aggregated stepwise head scores to {stepwise_path}")


if __name__ == "__main__":
    # Tons of defaults set, check out the LLMNeedleHaystackTester's init for more info
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--s_len', metavar='N', type=int, help='a number')
    parser.add_argument('-e', '--e_len', metavar='N', type=int, help='a number')
    parser.add_argument('--model_path', type=str, default=None, help='path to model')
    parser.add_argument('--model_name', type=str, default=None, help='name of model')
    parser.add_argument('--model_name_suffix', type=str, default=None, help='name of model')
    parser.add_argument('--model_provider', type=str, default="LLaMA", help='which model to use')
    args = parser.parse_args()
   
    model_name = args.model_path
    print(model_name)

    ht = LLMNeedleHaystackTester(model_name=model_name, 
                                 model_name_suffix=args.model_name_suffix,
                                 model_provider=args.model_provider,
                                 save_contexts=True,
                                 save_results=True,
                                 context_lengths_min=args.s_len,
                                 context_lengths_max=args.e_len,
                                 )

    ht.start_test(args)
