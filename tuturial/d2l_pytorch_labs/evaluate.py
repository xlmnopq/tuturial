"""在独立进程中恢复 best.pt，并对保存的测试划分重新评估。"""
import argparse
from pathlib import Path

from torch.utils.data import Subset
from torchvision.datasets import CIFAR10

from labkit.common import save_json
from labkit.data import (ReviewDataset, collate_reviews, image_transforms,
                         make_loader, read_jsonl)
from labkit.engine import run_epoch
from labkit.inference import restore_run
from labkit.plots import plot_confusion


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="auto")
    p.add_argument("--full-test", action="store_true", help="对官方完整测试集评估，包括 smoke 模型")
    args = p.parse_args()
    model, config, manifest, vocab, device = restore_run(args.run_dir, args.device)
    root = Path(args.data_dir)
    if config["task"] == "image":
        _, transform, _, _ = image_transforms(config["variant"], augment=False)
        source = CIFAR10(str(root / "cifar10"), train=False, download=False, transform=transform)
        dataset = source if args.full_test else Subset(source, manifest["test_indices"])
        collate = None
    else:
        source = read_jsonl(root / "imdb" / "test.jsonl")
        records = source if args.full_test else [source[i] for i in manifest["test_indices"]]
        dataset = ReviewDataset(records, vocab, config["max_length"])
        collate = collate_reviews
    loader = make_loader(dataset, args.batch_size, False, config["seed"], collate_fn=collate)
    metrics = run_epoch(model, loader, device)
    metrics["classes"] = manifest["classes"]
    metrics["full_test_requested"] = args.full_test
    save_json(Path(args.run_dir) / "reevaluation.json", metrics)
    fig = plot_confusion(metrics["confusion_matrix"], manifest["classes"],
                          Path(args.run_dir) / "reevaluation_confusion.png")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print("Test n=%d accuracy=%.4f macro_f1=%.4f" %
          (metrics["samples"], metrics["accuracy"], metrics["macro_f1"]))


if __name__ == "__main__":
    main()
