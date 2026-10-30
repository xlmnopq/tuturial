"""汇总真实运行结果。拒绝混用不同数据划分或 smoke / 完整实验。"""
import argparse
import json
from pathlib import Path

from labkit.common import write_csv


def summarize(run_dirs, output):
    rows, reference = [], None
    for directory in run_dirs:
        root = Path(directory)
        read = lambda name: json.loads((root / name).read_text(encoding="utf-8"))
        config, manifest, metrics = read("config.json"), read("data_manifest.json"), read("test_metrics.json")
        signature = {k: manifest.get(k) for k in ["dataset", "seed", "train_indices", "val_indices",
                                                   "test_indices", "max_length", "max_vocab", "min_freq",
                                                   "variant", "augment"]}
        if reference is None:
            reference = signature
        elif signature != reference:
            raise ValueError("数据划分或预处理不同，请分组比较: " + str(root))
        rows.append({"run": str(root), "model": config["model"], "dataset": manifest["dataset"],
                     "variant": config.get("variant", "word_level"),
                     "pretrained": config.get("pretrained", False),
                     "parameters": config["parameters"]["total"],
                     "train_samples": manifest["sizes"]["train"], "test_samples": metrics["samples"],
                     "test_scope": metrics["test_scope"], "accuracy": metrics["accuracy"],
                     "macro_f1": metrics["macro_f1"]})
    write_csv(output, rows)
    for row in rows:
        print("%s acc=%.4f F1=%.4f [%s]" %
              (row["model"], row["accuracy"], row["macro_f1"], row["test_scope"]))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output", default="runs/comparison.csv")
    args = parser.parse_args()
    summarize(args.runs, args.output)
