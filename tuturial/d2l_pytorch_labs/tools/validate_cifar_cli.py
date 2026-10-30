"""真实 CIFAR-10 的命令行训练与恢复检查，不下载预训练权重。"""
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).absolute().parents[1]
os.chdir(ROOT)
base = Path("runs") / datetime.now().strftime("cifar_cli_%Y%m%d_%H%M%S")


def run(*args):
    print("RUN", *map(str, args), flush=True)
    subprocess.run([sys.executable, *map(str, args)], check=True)


if __name__ == "__main__":
    for name in ["smallcnn", "alexnet", "vgg11", "resnet18"]:
        run("train_image.py", "--model", name, "--smoke", "--offline",
            "--batch-size", "16", "--run-dir", base / name)
    run("evaluate.py", "--run-dir", base / "resnet18", "--batch-size", "16")
    run("compare_runs.py", "--runs", base / "alexnet", base / "vgg11", base / "resnet18",
        "--output", base / "comparison.csv")
    (ROOT / "docs/cifar_cli_run_path.txt").write_text(str(base), encoding="utf-8")
    print("CIFAR CLASSIFICATION CLI CHECKS PASSED", flush=True)
