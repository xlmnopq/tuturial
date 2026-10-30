"""训练曲线、混淆矩阵和图像可视化。"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_history(history, path=None):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = [x["epoch"] for x in history]
    for split in ("train", "val"):
        axes[0].plot(epochs, [x[split + "_loss"] for x in history], label=split, marker="o")
        axes[1].plot(epochs, [x[split + "_accuracy"] for x in history], label=split, marker="o")
    for ax, name in zip(axes, ["Cross entropy", "Accuracy"]):
        ax.set(xlabel="Epoch", ylabel=name)
        ax.legend()
        ax.grid(alpha=0.25)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=140)
    return fig


def plot_confusion(matrix, classes, path=None):
    matrix = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(matrix, cmap="Blues")
    ax.set(xticks=range(len(classes)), yticks=range(len(classes)),
           xticklabels=classes, yticklabels=classes,
           xlabel="Predicted", ylabel="True label", title="Test confusion matrix")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black", fontsize=8)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=140)
    return fig


def show_images(images, labels, classes, mean=(0.5,) * 3, std=(0.5,) * 3,
                predictions=None, count=12):
    count = min(count, len(images))
    fig, axes = plt.subplots(1 + (count - 1) // 4, 4, figsize=(12, 3 * (1 + (count - 1) // 4)))
    for i, ax in enumerate(np.asarray(axes).reshape(-1)):
        ax.axis("off")
        if i >= count:
            continue
        image = images[i].detach().cpu().permute(1, 2, 0).numpy()
        image = np.clip(image * np.asarray(std) + np.asarray(mean), 0, 1)
        ax.imshow(image)
        label = classes[int(labels[i])]
        title = label if predictions is None else label + " / " + classes[int(predictions[i])]
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


def save_run_plots(history, metrics, run_dir):
    run_dir = Path(run_dir)
    fig = plot_history(history, run_dir / "learning_curves.png")
    plt.close(fig)
    fig = plot_confusion(metrics["confusion_matrix"], metrics["classes"], run_dir / "confusion_matrix.png")
    plt.close(fig)
