"""对两个真实数据集执行所有教学模型的最小训练、独立复测和预测。"""
import gc
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).absolute().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


def run(command):
    print("RUN", " ".join(map(str, command)), flush=True)
    subprocess.run([sys.executable, *map(str, command)], check=True)


def main():
    base = Path("runs") / datetime.now().strftime("validation_%Y%m%d_%H%M%S")
    for model in ["smallcnn", "alexnet", "vgg11", "resnet18"]:
        run(["train_image.py", "--model", model, "--smoke", "--offline",
             "--batch-size", "16", "--run-dir", base / model])
    for model in ["bilstm", "textcnn"]:
        run(["train_text.py", "--model", model, "--smoke", "--offline",
             "--batch-size", "16", "--run-dir", base / model])
    for model in ["vae", "gan"]:
        run(["train_" + model + ".py", "--smoke", "--offline",
             "--batch-size", "16", "--run-dir", base / model])
        run(["evaluate_generative.py", "--run-dir", base / model, "--batch-size", "16"])
        run(["generate.py", "--run-dir", base / model, "--num-samples", "16", "--seed", "2026"])
    run(["evaluate.py", "--run-dir", base / "resnet18"])
    run(["evaluate.py", "--run-dir", base / "textcnn"])
    run(["predict.py", "--run-dir", base / "textcnn", "--text", "A wonderful film with great acting."])
    from torchvision.datasets import CIFAR10
    ds = CIFAR10("data/cifar10", train=False, download=False)
    sample_path = base / "sample.png"
    ds[0][0].save(sample_path)
    run(["predict.py", "--run-dir", base / "resnet18", "--image", sample_path])
    run(["compare_runs.py", "--runs", base / "alexnet", base / "vgg11", base / "resnet18",
         "--output", base / "image_comparison.csv"])
    run(["compare_runs.py", "--runs", base / "bilstm", base / "textcnn",
         "--output", base / "text_comparison.csv"])
    print("ALL REAL-DATA SMOKE CHECKS PASSED:", base, flush=True)
    (ROOT / "docs" / "smoke_run_path.txt").write_text(str(base), encoding="utf-8")


if __name__ == "__main__":
    main()
