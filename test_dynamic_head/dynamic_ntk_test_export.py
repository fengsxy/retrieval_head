# %% Cell
import os

TARGET_GPU = os.environ.get("TARGET_GPU", "2")
os.environ["CUDA_VISIBLE_DEVICES"] = TARGET_GPU
print("CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])
print("Inside this process cuda:0 maps to physical GPU", TARGET_GPU)

# %% Cell
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from pprint import pprint
import json

import torch
from transformers import AutoTokenizer

from configuration_llada import LLaDAConfig
from modeling_llada import LLaDAModelLM
from mixed_rope_patch import apply_mixed_rope_patch

MODEL_PATH = Path("/data/ylong030/huggingface/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07").expanduser()
MODEL_PATH = Path("/data/ylong030/huggingface/hub/models--relaxe-system-lab--UltraLLaDA/snapshots/ultrallada").expanduser()

HEAD_SCORE_PATH = Path("../head_score/llada-block-2500.json").expanduser()
ROPE_SCALING_FACTOR: Optional[float] = 32
HEAD_SCORE_TOP_K = 16
HEAD_SCORE_THRESHOLD: Optional[float] = None  # 例如 0.5

DTYPE = torch.bfloat16
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

GEN_STEPS = 128
GEN_LENGTH = 128
BLOCK_LENGTH = 32
TEMPERATURE = 0.0
CFG_SCALE = 0.0
REMASKING = "low_confidence"  # or 'random'
MASK_TOKEN_ID = 126336

TEST_PROMPTS = [
    "The capital of France is",
    "Explain why dynamic head-score NTK scaling can help long-context retrieval.",
]

print(f"Using device: {DEVICE}")
print(f"Weights dir: {MODEL_PATH}")
print(f"Head scores: {HEAD_SCORE_PATH}")

# %% Cell
def load_head_scores(path: Path) -> List[Tuple[Tuple[int, int], float]]:
    text = path.read_text().strip()
    if not text:
        raise ValueError(f"Empty head score file: {path}")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        with path.open() as f:
            data = json.loads(f.readline())
    scored = []
    for key, values in data.items():
        try:
            layer_idx, head_idx = map(int, key.split('-'))
        except ValueError:
            continue
        if isinstance(values, Sequence):
            vals = [float(v) for v in values if v is not None]
            if not vals:
                continue
            score = sum(vals) / len(vals)
        else:
            score = float(values)
        scored.append(((layer_idx, head_idx), float(score)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored

def select_scaled_heads(
    scored_heads: List[Tuple[Tuple[int, int], float]],
    top_k: Optional[int],
    threshold: Optional[float] = None,
) -> Dict[int, set]:
    selected: Dict[int, set] = {}
    total = 0
    for (layer_idx, head_idx), score in scored_heads:
        if threshold is not None and score < threshold:
            break
        if top_k is not None and total >= top_k:
            break
        selected.setdefault(layer_idx, set()).add(head_idx)
        total += 1
    return selected

# %% Cell
config = LLaDAConfig.from_pretrained(str(MODEL_PATH))
config.use_cache = False  # diffusion decoding does not reuse KV cache
if ROPE_SCALING_FACTOR is not None:
    config.rope_scaling_factor = ROPE_SCALING_FACTOR

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

scaled_heads_dict: Dict[int, set] = {}
scored_heads: List[Tuple[Tuple[int, int], float]] = []
if HEAD_SCORE_PATH.exists():
    print("Loading head scores...")
    scored_heads = load_head_scores(HEAD_SCORE_PATH)
    selection = select_scaled_heads(
        scored_heads,
        top_k=HEAD_SCORE_TOP_K,
        threshold=HEAD_SCORE_THRESHOLD,
    )
    scaled_heads_dict.clear()
    scaled_heads_dict.update(selection)
    total_heads = sum(len(v) for v in scaled_heads_dict.values())
    print(f"Selected {total_heads} heads across {len(scaled_heads_dict)} layers")
    if scaled_heads_dict:
        preview = dict(list(scaled_heads_dict.items())[:3])
        pprint(preview)
    apply_mixed_rope_patch(
        model,
        scaling_factor=ROPE_SCALING_FACTOR or 1.0,
        scaled_heads_dict=scaled_heads_dict,
        verbose=True,
    )
else:
    print(f"⚠️  head score file not found: {HEAD_SCORE_PATH}")

print("Model dtype:", next(model.parameters()).dtype)
print("Configured rope scaling factor:", getattr(model.config, "rope_scaling_factor", "n/a"))
print("Patched scaled heads stored on config:", bool(getattr(model.config, "scaled_heads_dict", {})))

# %% Cell
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

# %% Cell
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
        block_mask = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask, steps_per_block)

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
                _, indices = torch.topk(confidence[row], k=quota)
                transfer_index[row, indices] = True
            x[transfer_index] = x0[transfer_index]

    return x

# %% Cell
def decode_prompt(
    prompt: str,
    steps: int = GEN_STEPS,
    gen_length: int = GEN_LENGTH,
    block_length: int = BLOCK_LENGTH,
    temperature: float = TEMPERATURE,
    cfg_scale: float = CFG_SCALE,
    remasking: str = REMASKING,
) -> str:
    encoded = tokenizer(prompt, return_tensors="pt")
    output_ids = llada_decode(
        model,
        encoded["input_ids"],
        steps=steps,
        gen_length=gen_length,
        block_length=block_length,
        temperature=temperature,
        cfg_scale=cfg_scale,
        remasking=remasking,
    )
    completion_ids = output_ids[:, encoded["input_ids"].shape[1]:]
    return tokenizer.decode(completion_ids[0], skip_special_tokens=True).strip()

def run_batch(prompts: List[str], **kwargs) -> None:
    for idx, prompt in enumerate(prompts, 1):
        print(f"Prompt {idx}: {prompt}")
        completion = decode_prompt(prompt, **kwargs)
        print(completion if completion else "[empty]")
        print("-" * 72)

# %% Cell
# 手动 prompt，可根据需要反复修改后运行本单元
custom_prompt ='''I attended to all the ghastly\nformalities, and the urbane undertaker proved that his staff were\nafflicted--or blessed--with something of his own obsequious suavity.\nEven the woman who performed the last offices for the dead remarked to\nme, in a confidential, brother-professional way, when she had come out\nfrom the death-chamber:--\n\n\"She makes a very beautiful corpse, sir. It's quite a privilege to\nattend on her. It's not too much to say that she will do credit to our\nestablishment!\"\n\nI noticed that Van Helsing never kept far away. This was possible from\nthe disordered state of things in the household. There were no relatives\nat hand; and as Arthur had to be back the next day to attend at his\nfather's funeral, we were unable to notify any one who should have been\nbidden. Under the circumstances, Van Helsing and I took it upon\nourselves to examine papers, etc. He insisted upon looking over Lucy's\npapers himself. I asked him why, for I feared that he, being a\nforeigner, might not be quite aware of English legal requirements, and\nso might in ignorance make some unnecessary trouble. He answered me:--\n\n\"I know; I know. You forget that I am a lawyer as well as a doctor. But\nthis is not altogether for the law. You knew that, when you avoided the\ncoroner. I have more than him to avoid. There may be papers more--such\nas this.\"\n\nAs he spoke he took from his pocket-book the memorandum which had been\nin Lucy's breast, and which she had torn in her sleep.\n\n\"When you find anything of the solicitor who is for the late Mrs.\nWestenra, seal all her papers, and write him to-night. For me, I watch\nhere in the room and in Miss Lucy's old room all night, and I myself\nsearch for what may be. It is not well that her very thoughts go into\nthe hands of strangers.\"\n\nI went on with my part of the work, and in another half hour had found\nthe name and address of Mrs. Westenra's solicitor and had written to\nhim. All the poor lady's papers were in order; explicit directions\nregarding the place of burial were given. I had hardly sealed the\nletter, when, to my surprise, Van Helsing walked into the room,\nsaying:--\n\n\"Can I help you, friend John? I am free, and if I may, my service is to\nyou.\"\n\n\"Have you got what you looked for?\" I asked, to which he replied:--\n\n\"I did not look for any specific thing. I only hoped to find, and find I\nhave, all that there was--only some letters and a few memoranda, and a\ndiary new begun. But I have them here, and we shall for the present say\nnothing of them. I shall see that poor lad to-morrow evening, and, with\nhis sanction, I shall use some.\"\n\nWhen we had finished the work in hand, he said to me:--\n\n\"And now, friend John, I think we may to bed. We want sleep, both you\nand I, and rest to recuperate. To-morrow we shall have much to do, but\nfor the to-night there is no need of us. Alas!\"\n\nBefore turning in we went to look at poor Lucy. The undertaker had\ncertainly done his work well, for the room was turned into a small\n_chapelle ardente_. There was a wilderness of beautiful white flowers,\nand death was made as little repulsive as might be. The end of the\nwinding-sheet was laid over the face; when the Professor bent over and\nturned it gently back, we both started at the beauty before us, the tall\nwax candles showing a sufficient light to note it well. All Lucy's\nloveliness had come back to her in death, and the hours that had passed,\ninstead of leaving traces of \"decay's effacing fingers,\" had but\nrestored the beauty of life, till positively I could not believe my eyes\nthat I was looking at a corpse.\n\nThe Professor looked sternly grave. He had not loved her as I had, and\nthere was no need for tears in his eyes. He said to me: \"Remain till I\nreturn,\" and left the room. He came back with a handful of wild garlic\nfrom the box waiting in the hall, but which had not been opened, and\nplaced the flowers amongst the others on and around the bed. Then he\ntook from his neck, inside his collar, a little gold crucifix, and\nplaced it over the mouth. He restored the sheet to its place, and we\ncame away.\n\nI was undressing in my own room, when, with a premonitory tap at the\ndoor, he entered, and at once began to speak:--\n\n\"To-morrow I want you to bring me, before night, a set of post-mortem\nknives.\"\n\n\"Must we make an autopsy?\" I asked.\n\n\"Yes and no. I want to operate, but not as you think. Let me tell you\nnow, but not a word to another. I want to cut off her head and take out\nher heart. Ah! you a surgeon, and so shocked! You, whom I have seen with\nno tremble of hand or heart, do operations of life and death that make\nthe rest shudder. Oh, but I must not forget, my dear friend John, that\nyou loved her; and I have not forgotten it, for it is I that shall\noperate, and you must only help. I would like to do it to-night, but for\nArthur I must not; he will be free after his father's funeral to-morrow,\nand he will want to see her--to see _it_. Then, when she is coffined\nready for the next day, you and I shall come when all sleep. We shall\nunscrew the coffin-lid, and shall do our operation: and then replace\nall, so that none know, save we alone.\"\n\n\"But why do it at all? The girl is dead. Why mutilate her poor body\nwithout need? And if there is no necessity for a post-mortem and nothing\nto gain by it--no good to her, to us, to science, to human\nknowledge--why do it? Without such it is monstrous.\"\n\nFor answer he put his hand on my shoulder, and said, with infinite\ntenderness:--\n\n\"Friend John, I pity your poor bleeding heart; and I love you the more\nbecause it does so bleed. If I could, I would take on myself the burden\nthat you do bear. But there are things that you know not, but that you\nshall know, and bless me for knowing, though they are not pleasant\nthings. John, my child, you have been my friend now many years, and yet\ndid you ever know me to do any without good cause? I may err--I am but\nman; but I believe in all I do. Was it not for these causes that you\nsend for me when the great trouble came? Yes! Were you not amazed, nay\nhorrified, when I would not let Arthur kiss his love--though she was\ndying--and snatched him away by all my strength? Yes! And yet you saw\nhow she thanked me, with her so beautiful dying eyes, her voice, too, so\nweak, and she kiss my rough old hand and bless me? Yes! And did you not\nhear me swear promise to her, that so she closed her eyes grateful? Yes!\n\n\"Well, I have good reason now for all I want to do. You have for many\nyears trust me; you have believe me weeks past, when there be things so\nstrange that you might have well doubt. Believe me yet a little, friend\nJohn. If you trust me not, then I must tell what I think; and that is\nnot perhaps well. And if I work--as work I shall, no matter trust or no\ntrust--without my friend trust in me, I work with heavy heart and feel,\noh! so lonely when I want all help and courage that may be!\" He paused a\nmoment and went on solemnly: \"Friend John, there are strange and\nterrible days before us. Let us not be two, but one, that so we work to\na good end. Will you not have faith in me?\"\n\nI took his hand, and promised him. I held my door open as he went away,\nand watched him go into his room and close the door. As I stood without\nmoving, I saw one of the maids pass silently along the passage--she had\nher back towards me, so did not see me--and go into the room where Lucy\nlay. The sight touched me. Devotion is so rare, and we are so grateful\nto those who show it unasked to those we love. Here was a poor girl\nputting aside the terrors which she naturally had of death to go watch\nalone by the bier of the mistress whom she loved, so that the poor clay\nmight not be lonely till laid to eternal rest....\n\n       *       *       *       *       *\n\nI must have slept long and soundly, for it was broad daylight when Van\nHelsing waked me by coming into my room. He came over to my bedside and\nsaid:--\n\n\"You need not trouble about the knives; we shall not do it.\"\n\n\"Why not?\" I asked. For his solemnity of the night before had greatly\nimpressed me.\n\n\"Because,\" he said sternly, \"it is too late--or too early. See!\" Here he\nheld up the little golden crucMr Green is disliked by everyone because he is a mean person and also he can't ride a horse or dive a car.'''
custom_prompt= '''"Below is a numbered list of words. In these words, some appear more often than others. Memorize the ones that appear most often.\n1. reminiscent 2. lambkin 3. reminiscent 4. ripe 5. priesthood 6. yurt 7. chiffonier 8. yesterday 9. resolution 10. resolution 11. application 12. trot 13. pail 14. resolution 15. muscle 16. lambkin 17. appliance 18. reminiscent 19. trot 20. coconut 21. application 22. landform 23. minimalism 24. priesthood 25. chiffonier 26. fly 27. negotiate 28. daikon 29. resolution 30. avenue 31. appliance 32. yurt 33. reminiscent 34. priesthood 35. scattered 36. reminiscent 37. syrup 38. isolation 39. priesthood 40. bibliography 41. pail 42. penicillin 43. lye 44. ischemia 45. emergency 46. pail 47. yurt 48. hypothermia 49. priesthood 50. stand 51. yurt 52. isolation 53. trot 54. negotiate 55. stand 56. lament 57. lambkin 58. bay 59. pocketbook 60. resolution 61. lament 62. priesthood 63. bay 64. lye 65. lambkin 66. pattern 67. scattered 68. lambkin 69. trot 70. bay 71. lye 72. stand 73. coconut 74. lambkin 75. lye 76. pattern 77. tow 78. scattered 79. hypothermia 80. resolution 81. bay 82. yesterday 83. isolation 84. modify 85. resolution 86. resolution 87. self 88. bondsman 89. reminiscent 90. daikon 91. self 92. bullet 93. syrup 94. penicillin 95. lambkin 96. lye 97. minimalism 98. syrup 99. minimalism 100. penicillin 101. bondsman 102. self 103. muscle 104. landform 105. bay 106. tow 107. pocketbook 108. muscle 109. resolution 110. bay 111. syrup 112. ripe 113. daikon 114. lye 115. reminiscent 116. yesterday 117. lye 118. lambkin 119. trot 120. lament 121. trot 122. daikon 123. appliance 124. yurt 125. avenue 126. syrup 127. bullet 128. ripe 129. chiffonier 130. trot 131. coconut 132. hypothermia 133. lambkin 134. bay 135. tuxedo 136. daikon 137. lye 138. reminiscent 139. trot 140. priesthood 141. yurt 142. syrup 143. modify 144. ischemia 145. syrup 146. daikon 147. bay 148. syrup 149. lambkin 150. daikon 151. tuxedo 152. pocketbook 153. yurt 154. emergency 155. lye 156. tow 157. daikon 158. ischemia 159. priesthood 160. syrup 161. syrup 162. bibliography 163. tuxedo 164. yurt 165. modify 166. bibliography 167. avenue 168. priesthood 169. fly 170. bondsman 171. negotiate 172. application 173. lye 174. trot 175. resolution 176. daikon 177. bay 178. landform 179. emergency 180. reminiscent 181. pattern 182. fly 183. yurt 184. trot 185. daikon 186. reminiscent 187. priesthood 188. yurt 189. bay 190. bullet\nQuestion: What are the 10 most common words in the above list? Answer: The top 10 words that appear most often in the list are:1. priesthood 2. syrup 3. reminiscent 4. resolution 5. lambkin 6. trot 7. yurt 8. daikon 9. bay 10. lye\nBelow is a numbered list of words. In these words, some appear more often than others. Memorize the ones that appear most often.\n1. exotic 2. bed 3. tulip 4. scold 5. glass 6. ferryboat 7. organ 8. homeownership 9. socialism 10. confidentiality 11. instance 12. woolens 13. overthrow 14. horizon 15. confidentiality 16. foretell 17. rations 18. tomorrow 19. eggnog 20. rations 21. reproduce 22. socialism 23. fireplace 24. outlet 25. ferryboat 26. pith 27. quotation 28. actor 29. quotation 30. colon 31. ferryboat 32. foretell 33. heartbeat 34. socialism 35. heartbeat 36. fireplace 37. compete 38. accordion 39. fireplace 40. medal 41. ferryboat 42. ban 43. confidentiality 44. tissue 45. opera 46. fireplace 47. gloom 48. fireplace 49. procurement 50. do 51. perennial 52. confidentiality 53. rations 54. fireplace 55. marines 56. socialism 57. loophole 58. authenticity 59. vague 60. ferryboat 61. confidentiality 62. neonate 63. heartbeat 64. rations 65. escalator 66. drill 67. actor 68. stranger 69. kind 70. heartbeat 71. granny 72. granny 73. heartbeat 74. consul 75. fireplace 76. assurance 77. confidentiality 78. gloom 79. socialism 80. make 81. examination 82. confidentiality 83. ferryboat 84. confidentiality 85. baseline 86. fate 87. ferryboat 88. tissue 89. narrow 90. ferryboat 91. foretell 92. woolens 93. bulldozer 94. hair 95. actor 96. actor 97. coherent 98. lieu 99. front 100. kind 101. socialism 102. rations 103. tissue 104. island 105. palm 106. cycle 107. tissue 108. material 109. actor 110. kind 111. kind 112. narrow 113. ferryboat 114. socialism 115. ferryboat 116. trait 117. actor 118. foretell 119. stealth 120. pith 121. rations 122. revolution 123. do 124. lily 125. socialism 126. socialism 127. fireplace 128. tissue 129. foretell 130. actor 131. feedback 132. think 133. butane 134. disruption 135. methane 136. purity 137. medal 138. rations 139. adobe 140. foretell 141. socialism 142. meteor 143. instance 144. deranged 145. carving 146. island 147. tulip 148. tissue 149. galley 150. confidentiality 151. allowance 152. ferryboat 153. foretell 154. scold 155. confidentiality 156. confidentiality 157. actor 158. bicycle 159. trait 160. advertising 161. socialism 162. butter 163. tissue 164. make 165. tulip 166. socialism 167. front 168. lily 169. carnival 170. butter 171. confidentiality 172. ferryboat 173. homeownership 174. confidentiality 175. foretell 176. exhaust 177. crawl 178. disillusioned 179. actor 180. characteristic 181. kiwi 182. winery 183. clapboard 184. heartbeat 185. outlet 186. fireplace 187. designation 188. foretell 189. kind 190. tissue 191. confidentiality 192. overthrow 193. pickaxe 194. fireplace 195. fireplace 196. fireplace 197. tissue 198. heartbeat 199. purity 200. confidentiality 201. terrorism 202. carving 203. tower 204. tissue 205. actor 206. tissue 207. kind 208. rations 209. actor 210. colony 211. perennial 212. terrorism 213. exotic 214. trafficker 215. reality 216. trait 217. tissue 218. ferryboat 219. exotic 220. heartbeat 221. deranged 222. kind 223. pastry 224. eggnog 225. sustainment 226. actor 227. heartbeat 228. confidentiality 229. ferryboat 230. kind 231. tissue 232. tissue 233. disillusioned 234. kind 235. abbey 236. winery 237. examination 238. accordion 239. kind 240. socialism 241. ban 242. turmeric 243. lieu 244. tissue 245. ferryboat 246. bribery 247. ferryboat 248. kind 249. kind 250. consul 251. merit 252. psychedelic 253. escalator 254. meteor 255. heartbeat 256. colony 257. reproduce 258. instance 259. socialism 260. rations 261. heartbeat 262. detainment 263. pickaxe 264. heartbeat 265. confidentiality 266. confidentiality 267. kind 268. palm 269. fireplace 270. rations 271. clapboard 272. psychedelic 273. fireplace 274. heartbeat 275. fireplace 276. stranger 277. confidentiality 278. authenticity 279. outlaw 280. mill 281. methane 282. fireplace 283. actor 284. foretell 285. confidentiality 286. rations 287. socialism 288. socialism 289. hair 290. kind 291. retention 292. ferryboat 293. airforce 294. compete 295. butane 296. outfit 297. kind 298. foretell 299. socialism 300. heartbeat 301. confidentiality 302. procurement 303. think 304. tissue 305. reality 306. chair 307. eggnog 308. foretell 309. bicycle 310. outfit 311. tissue 312. chair 313. scold 314. methane 315. reproduce 316. foretell 317. tissue 318. palm 319. retention 320. lie 321. meteor 322. rations 323. heartbeat 324. compete 325. recession 326. tissue 327. trafficker 328. socialism 329. detainment 330. detainment 331. foretell 332. foretell 333. perennial 334. heartbeat 335. heartbeat 336. ban 337. feedback 338. galley 339. admire 340. consul 341. heartbeat 342. outlaw 343. rations 344. pastry 345. winery 346. fireplace 347. exhaust 348. actor 349. organ 350. virginal 351. fireplace 352. purity 353. socialism 354. socialism 355. kiwi 356. describe 357. lie 358. horizon 359. rose 360. actor 361. confidentiality 362. carving 363. actor 364. abbey 365. opera 366. pickaxe 367. gloom 368. heartbeat 369. carnival 370. baseline 371. confidentiality 372. disruption 373. fireplace 374. stranger 375. glass 376. actor 377. foretell 378. tissue 379. bed 380. characteristic 381. trafficker 382. drill 383. recession 384. heartbeat 385. tissue 386. describe 387. foretell 388. horizon 389. tower 390. heartbeat 391. assurance 392. socialism 393. confidentiality 394. material 395. rations 396. ferryboat 397. fireplace 398. ferryboat 399. reality 400. tower 401. hair 402. heartbeat 403. stinger 404. escalator 405. bicycle 406. foretell 407. kind 408. adobe 409. heartbeat 410. actor 411. rations 412. designation 413. outlaw 414. front 415. bulldozer 416. actor 417. kind 418. adobe 419. write 420. tomorrow 421. kind 422. crawl 423. tissue 424. kind 425. bed 426. fireplace 427. actor 428. foretell 429. foretell 430. rations 431. write 432. outfit 433. describe 434. virginal 435. rations 436. allowance 437. disruption 438. ferryboat 439. do 440. stinger 441. island 442. glass 443. confidentiality 444. butane 445. disillusioned 446. adult 447. fireplace 448. rations 449. cycle 450. kind 451. pastry 452. designation 453. rations 454. drill 455. ferryboat 456. outlet 457. fireplace 458. opera 459. confidentiality 460. deranged 461. kind 462. tissue 463. tissue 464. neonate 465. kind 466. heartbeat 467. ferryboat 468. socialism 469. accordion 470. think 471. airforce 472. fireplace 473. bribery 474. heartbeat 475. airforce 476. fireplace 477. assurance 478. tissue 479. turmeric 480. psychedelic 481. bulldozer 482. admire 483. stealth 484. ferryboat 485. advertising 486. foretell 487. confidentiality 488. foretell 489. write 490. baseline 491. carnival 492. marines 493. ferryboat 494. marines 495. stealth 496. rations 497. medal 498. heartbeat 499. foretell 500. revolution 501. actor 502. vague 503. rations 504. neonate 505. merit 506. adult 507. allowance 508. foretell 509. ferryboat 510. cartload 511. ferryboat 512. quotation 513. revolution 514. examination 515. virginal 516. socialism 517. foretell 518. socialism 519. adult 520. terrorism 521. rose 522. fireplace 523. clapboard 524. granny 525. actor 526. ferryboat 527. confidentiality 528. kind 529. material 530. crawl 531. foretell 532. characteristic 533. sustainment 534. lie 535. chair 536. heartbeat 537. ferryboat 538. abbey 539. foretell 540. actor 541. woolens 542. tomorrow 543. exhaust 544. socialism 545. loophole 546. actor 547. socialism 548. rations 549. fate 550. lily 551. admire 552. rations 553. socialism 554. bribery 555. rations 556. actor 557. colony 558. cartload 559. socialism 560. lieu 561. feedback 562. fireplace 563. kind 564. kind 565. rations 566. cartload 567. mill 568. actor 569. colon 570. coherent 571. mill 572. colon 573. kind 574. rose 575. foretell 576. actor 577. kind 578. actor 579. foretell 580. actor 581. homeownership 582. fate 583. butter 584. tissue 585. vague 586. socialism 587. recession 588. sustainment 589. rations 590. tissue 591. authenticity 592. turmeric 593. heartbeat 594. advertising 595. rations 596. overthrow 597. procurement 598. tissue 599. heartbeat 600. kiwi 601. stinger 602. retention 603. kind 604. actor 605. rations 606. make 607. pith 608. fireplace 609. galley 610. organ 611. kind 612. coherent 613. socialism 614. cycle 615. confidentiality 616. kind 617. fireplace 618. tissue 619. ferryboat 620. loophole 621. narrow 622. heartbeat 623. merit 624. rations 625. ferryboat 626. confidentiality 627. tissue 628. foretell 629. fireplace 630. rations\nQuestion: What are the 10 most common words in the above list? Answer: The top 10 words that appear most often in the list are:\n\n"'''
print(decode_prompt(custom_prompt, steps=GEN_STEPS, gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH))

# %% Cell
TOP_K_SWEEP = [0, 16, 32, 128, 256, 512]
print(f"Running sweep for head-score top-k values: {TOP_K_SWEEP}")

def update_scaled_heads(top_k: Optional[int], threshold: Optional[float] = HEAD_SCORE_THRESHOLD) -> None:
    if not scored_heads:
        raise RuntimeError("Head scores are not loaded; cannot update head scaling.")
    selection = select_scaled_heads(
        scored_heads,
        top_k=top_k,
        threshold=threshold,
    )
    scaled_heads_dict.clear()
    scaled_heads_dict.update(selection)
    model.config.scaled_heads_dict = scaled_heads_dict
    model.config.head_score_top_k = top_k or 0
    model.config.head_score_threshold = threshold
    total_heads = sum(len(v) for v in scaled_heads_dict.values())
    print(f"→ Updated scaled heads: top_k={top_k}, layers={len(scaled_heads_dict)}, total_heads={total_heads}")

def sweep_prompt(prompt: str, top_k_values: Sequence[int], **decode_kwargs) -> Dict[int, str]:
    results: Dict[int, str] = {}
    for top_k in top_k_values:
        print(f"\n=== HEAD_SCORE_TOP_K={top_k} ===")
        update_scaled_heads(top_k)
        completion = decode_prompt(prompt, **decode_kwargs)
        results[top_k] = completion
        print(completion if completion else "[empty]")
        print("-" * 72)
    return results

head_sweep_results = sweep_prompt(
    custom_prompt,
    TOP_K_SWEEP,
    steps=GEN_STEPS,
    gen_length=GEN_LENGTH,
    block_length=BLOCK_LENGTH,
    temperature=TEMPERATURE,
    cfg_scale=CFG_SCALE,
    remasking=REMASKING,
)

# %% Cell

