import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


GROUP_SIZE = 8
STEPWISE_DIR = Path("head_score")
OUTPUT_DIR = Path("viz")
OUTPUT_DIR.mkdir(exist_ok=True)


def _load_stepwise(path: Path) -> Tuple[List[str], np.ndarray]:
    """Return ordered step names and [num_steps, layers, heads] score tensor."""
    if path.stat().st_size == 0:
        raise ValueError(f"Stepwise file is empty: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw: Dict[str, Dict[str, List[float]]] = json.load(f)

    ordered_steps = sorted(raw.items(), key=lambda item: int(item[0].split("_")[-1]))
    if not ordered_steps:
        raise ValueError(f"No step entries found in {path}")

    sample_heads = ordered_steps[0][1]
    if not sample_heads:
        raise ValueError(f"No head entries found in first step of {path}")

    layer_ids = []
    head_ids = []
    for key in sample_heads.keys():
        layer, head = key.split("-")
        layer_ids.append(int(layer))
        head_ids.append(int(head))
    num_layers = max(layer_ids) + 1
    num_heads = max(head_ids) + 1

    tensors: List[np.ndarray] = []
    step_names: List[str] = []
    for step_name, head_scores in ordered_steps:
        matrix = np.zeros((num_layers, num_heads), dtype=np.float64)
        for head_key, values in head_scores.items():
            layer, head = (int(part) for part in head_key.split("-"))
            matrix[layer, head] = float(np.mean(values))
        tensors.append(matrix)
        step_names.append(step_name)
    return step_names, np.stack(tensors)


def _pearson(vec_i: np.ndarray, vec_j: np.ndarray) -> float:
    """Compute Pearson correlation with safeguards."""
    centered_i = vec_i - vec_i.mean()
    centered_j = vec_j - vec_j.mean()
    denom = np.linalg.norm(centered_i) * np.linalg.norm(centered_j)
    if math.isclose(denom, 0.0):
        return 0.0
    return float(np.dot(centered_i, centered_j) / denom)


def _group_average(step_tensor: np.ndarray, step_names: List[str]) -> Tuple[List[str], List[np.ndarray]]:
    """Aggregate steps into groups of GROUP_SIZE."""
    num_steps = step_tensor.shape[0]
    groups: List[np.ndarray] = []
    labels: List[str] = []

    for start in range(0, num_steps, GROUP_SIZE):
        end = min(start + GROUP_SIZE, num_steps)
        group_tensor = step_tensor[start:end]
        group_avg = group_tensor.mean(axis=0)
        groups.append(group_avg)
        labels.append(f"{start + 1}-{end}")
    return labels, groups


def _plot_group_heatmaps(model_name: str, group_labels: List[str], group_tensors: List[np.ndarray]) -> Path:
    """Plot group heatmaps and return output path."""
    num_groups = len(group_tensors)
    rows = math.ceil(num_groups / 2)
    cols = 2 if num_groups > 1 else 1

    data_stack = np.array([tensor for tensor in group_tensors])
    vmin = data_stack.min()
    vmax = data_stack.max()

    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4 * rows), layout="constrained")
    axes = np.array(axes).reshape(rows, cols)

    for idx, (label, tensor) in enumerate(zip(group_labels, group_tensors)):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        im = ax.imshow(tensor.T, origin="upper", aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"Steps {label}")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Head")
    # hide any unused subplots
    for idx in range(len(group_tensors), rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis("off")

    # single colorbar
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.9)
    cbar.set_label("Avg Retrieval Score", rotation=270, labelpad=15)

    output_path = OUTPUT_DIR / f"{model_name}_step_groups.png"
    fig.suptitle(f"{model_name} Retrieval Head Heatmaps by Step Groups", fontsize=14)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def _plot_similarity_matrix(model_name: str, group_labels: List[str], group_tensors: List[np.ndarray]) -> Path:
    """Plot Pearson similarity between group heatmaps."""
    vectors = [tensor.flatten() for tensor in group_tensors]
    size = len(vectors)
    matrix = np.zeros((size, size), dtype=np.float64)
    for i in range(size):
        for j in range(size):
            matrix[i, j] = _pearson(vectors[i], vectors[j])

    fig, ax = plt.subplots(figsize=(5.5, 4.5), layout="constrained")
    im = ax.imshow(matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(size), labels=group_labels, rotation=25, ha="right")
    ax.set_yticks(range(size), labels=group_labels)
    ax.set_title(f"{model_name} Step Group Similarity (Pearson)")

    for i in range(size):
        for j in range(size):
            value = matrix[i, j]
            color = "white" if abs(value) > 0.5 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=color)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Pearson Correlation", rotation=270, labelpad=15)

    output_path = OUTPUT_DIR / f"{model_name}_step_similarity.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"{model_name} similarity matrix:\n{matrix}")
    return output_path


def main() -> None:
    stepwise_paths = sorted(STEPWISE_DIR.glob("*_stepwise.json"))
    if not stepwise_paths:
        raise FileNotFoundError("No *_stepwise.json files found in head_score/")

    for path in stepwise_paths:
        model_name = path.stem.replace("_stepwise", "")
        try:
            step_names, step_tensor = _load_stepwise(path)
        except ValueError as err:
            print(f"Skipping {path.name}: {err}")
            continue

        group_labels, group_tensors = _group_average(step_tensor, step_names)
        heatmap_path = _plot_group_heatmaps(model_name, group_labels, group_tensors)
        similarity_path = _plot_similarity_matrix(model_name, group_labels, group_tensors)
        print(f"{model_name}: heatmaps saved to {heatmap_path.resolve()}")
        print(f"{model_name}: similarity matrix saved to {similarity_path.resolve()}")


if __name__ == "__main__":
    main()

