import torch

from transformers import AutoTokenizer

from modeling_dream import DreamModel
from generation_utils import DreamGenerationConfig


def build_generation_config(model: DreamModel, steps: int, max_new_tokens: int) -> DreamGenerationConfig:
    """
    Create a fresh diffusion generation config so we do not mutate the model default.
    """
    generation_config = DreamGenerationConfig.from_model_config(model.config)
    generation_config.steps = steps
    generation_config.max_new_tokens = max_new_tokens
    generation_config.temperature = 0.0
    generation_config.alg = "origin"
    generation_config.return_dict_in_generate = False
    return generation_config


@torch.no_grad()
def diffusion_generate_response(
    model: DreamModel,
    tokenizer: AutoTokenizer,
    prompt: str,
    steps: int = 128,
    max_new_tokens: int = 128,
) -> str:
    """
    Run Dream diffusion generation using the sampler defined in generation_utils.py.
    """
    device = model.device
    messages = [{"role": "user", "content": prompt}]
    chat_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(chat_prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    generation_config = build_generation_config(model, steps=steps, max_new_tokens=max_new_tokens)
    sequences = model.diffusion_generate(
        inputs=input_ids,
        attention_mask=attention_mask,
        generation_config=generation_config,
    )

    generated_tokens = sequences[:, input_ids.shape[-1] :]
    text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return text.strip()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "Dream-org/Dream-v0-Instruct-7B"
    load_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = DreamModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=load_dtype,
    ).to(device).eval()

    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"
    response = diffusion_generate_response(model, tokenizer, prompt, steps=32, max_new_tokens=128)
    print(response)


if __name__ == "__main__":
    main()
