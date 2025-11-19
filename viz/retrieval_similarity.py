import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr

def load_heatmap(json_path: Path) -> np.ndarray:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    head_score_list = [([int(ll) for ll in k.split("-")], np.mean(v)) for k, v in data.items()]
    num_layers = max(lh[0] for lh, _ in head_score_list) + 1
    num_heads = max(lh[1] for lh, _ in head_score_list) + 1
    heatmap = np.full((num_layers, num_heads), np.nan)
    for (lh, score) in head_score_list:
        layer, head = lh
        heatmap[layer, head] = score
    return heatmap

def flatten_heatmap(h: np.ndarray) -> np.ndarray:
    return np.nan_to_num(h, nan=0.0).flatten()

# 模型路径
model_paths = {
    "qwen2.5": Path("head_score/qwen2.5.json"),
    "Qwen1.5-14B": Path("head_score/Qwen1.5-14B.json"),
     "Qwen1.5-14B chat":Path("head_score/Qwen1.5-14B-Chat.json"),
    "Dream": Path("head_score/dream.json"),
    "LLaDA-AR": Path("head_score/llada-2500.json"),
    "LLaDA-block": Path("head_score/llada-block-2500.json"),
}

heatmaps = {name: load_heatmap(p) for name, p in model_paths.items()}
min_layers = min(h.shape[0] for h in heatmaps.values())
min_heads = min(h.shape[1] for h in heatmaps.values())
heatmaps = {n: h[:min_layers, :min_heads] for n, h in heatmaps.items()}

# 计算 Spearman 排序相关矩阵
names = list(heatmaps.keys())
n = len(names)
similarity = np.zeros((n, n))
for i, ni in enumerate(names):
    for j, nj in enumerate(names):
        hi = flatten_heatmap(heatmaps[ni])
        hj = flatten_heatmap(heatmaps[nj])
        corr, _ = spearmanr(hi, hj)  # 关键变化：使用 Spearman 秩相关
        similarity[i, j] = corr

# 画图
plt.figure(figsize=(6, 5))
sns.heatmap(
    similarity, annot=True, fmt=".2f",
    cmap="magma", vmin=-1, vmax=1,
    cbar_kws={"label": "Spearman Rank Correlation"},
    xticklabels=names, yticklabels=names,
    linewidths=0.3, square=True
)
plt.title("Rank-Based Heatmap Structural Similarity Across Models", fontsize=11, pad=10)
plt.tight_layout()
plt.savefig("viz/heatmap_rank_similarity.png", dpi=300)
plt.show()
