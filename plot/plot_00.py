"""
plot_00.py

Plot confusion matrix from eval_00 saved CSV.

This script does NOT run model inference.
It only reads:
  results/eval_v2/<dataset>/eval_00/confusion_matrix.csv

Outputs:
  plot/results/eval_00/<dataset>/confusion_matrix.png
  plot/results/eval_00/<dataset>/confusion_matrix.pdf
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot eval_00 confusion matrix")
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--dataset", default=None, type=str)
    parser.add_argument("--output-root", default="plot/results/eval_00", type=str)
    parser.add_argument("--normalize", action="store_true")
    return parser.parse_args()


def load_confusion_matrix(path: Path) -> np.ndarray:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pred_cols = [k for k in row.keys() if k.startswith("pred_")]
            pred_cols = sorted(pred_cols, key=lambda x: int(x.split("_")[1]))
            rows.append([int(row[c]) for c in pred_cols])
    return np.asarray(rows, dtype=np.float64)


def plot_confusion(matrix: np.ndarray, output_dir: Path, normalize: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    data = matrix.copy()
    title = "Confusion matrix"

    if normalize:
        denom = data.sum(axis=1, keepdims=True)
        denom[denom == 0] = 1
        data = 100.0 * data / denom
        title = "Normalized confusion matrix"

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(data)

    num_classes = data.shape[0]
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Percent" if normalize else "Count")

    for i in range(num_classes):
        for j in range(num_classes):
            value = data[i, j]
            label = f"{value:.1f}" if normalize else f"{int(value)}"
            ax.text(j, i, label, ha="center", va="center", fontsize=7)

    fig.tight_layout()

    suffix = "_normalized" if normalize else ""
    fig.savefig(output_dir / f"confusion_matrix{suffix}.png", dpi=300)
    fig.savefig(output_dir / f"confusion_matrix{suffix}.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input)
    dataset = args.dataset or input_dir.parent.name
    output_dir = Path(args.output_root) / dataset

    matrix = load_confusion_matrix(input_dir / "confusion_matrix.csv")
    plot_confusion(matrix, output_dir, normalize=args.normalize)

    print(f"[plot_00] saved to {output_dir}")


if __name__ == "__main__":
    main()