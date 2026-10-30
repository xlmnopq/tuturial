"""随机种子、设备选择、固定数据划分和结果文件。"""
from __future__ import annotations

import csv
import hashlib
import inspect
import json
import os
import random
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed=42):
    # 小批量 GPU 课程避免 CPU 线程过多带来的调度开销，可用环境变量覆盖。
    threads = int(os.environ.get("LAB_NUM_THREADS", "4"))
    if threads < 1:
        raise ValueError("LAB_NUM_THREADS 必须为正整数")
    torch.set_num_threads(threads)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def choose_device(name="auto"):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("未检测到 CUDA。请安装匹配的 PyTorch，或使用 --device cpu。")
    return device


def seed_worker(worker_id):
    # DataLoader 子进程需要独立且可复现的 Python / NumPy 随机状态。
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def label_fingerprint(labels):
    return hashlib.sha256(np.asarray(labels, dtype=np.int64).tobytes()).hexdigest()


def stratified_split(labels, val_fraction=0.1, seed=42):
    """仅对官方训练集划分 train / val，保持每个类别的大致比例。"""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction 必须位于 (0, 1)")
    labels = np.asarray(labels)
    rng = np.random.RandomState(seed)
    train, val = [], []
    for label in np.unique(labels):
        ids = np.flatnonzero(labels == label)
        if len(ids) < 2:
            raise ValueError("每类至少需要两个样本才能划分训练集和验证集")
        ids = rng.permutation(ids)
        n_val = max(1, min(len(ids) - 1, round(len(ids) * val_fraction)))
        val.extend(ids[:n_val].tolist())
        train.extend(ids[n_val:].tolist())
    return sorted(train), sorted(val)


def fixed_split(labels, path, seed=42, val_fraction=0.1):
    path = Path(path)
    fingerprint = label_fingerprint(labels)
    if path.exists():
        obj = json.loads(path.read_text(encoding="utf-8"))
        if (obj["seed"] != seed or obj["val_fraction"] != val_fraction
                or obj["label_fingerprint"] != fingerprint):
            raise ValueError("缓存的数据划分与当前数据/参数不同，请指定新的划分文件。")
        train, val = obj["train"], obj["val"]
        if (len(set(train) | set(val)) != len(labels)
                or set(train) & set(val)
                or sorted(train + val) != list(range(len(labels)))):
            raise ValueError("数据划分文件损坏")
        return train, val
    train, val = stratified_split(labels, val_fraction, seed)
    save_json(path, {"seed": seed, "val_fraction": val_fraction,
                     "label_fingerprint": fingerprint, "train": train, "val": val})
    return train, val


def balanced_limit(indices, labels, limit, seed=42):
    """分层选取小样本；limit=0 表示使用全部数据。"""
    if limit <= 0 or limit >= len(indices):
        return list(indices)
    rng = np.random.RandomState(seed)
    groups = []
    for label in sorted(set(labels[i] for i in indices)):
        ids = [i for i in indices if labels[i] == label]
        groups.append(rng.permutation(ids).tolist())
    # 轮流从各类抽一个，避免按文件名截取导致类别偏置。
    result, offset = [], 0
    while len(result) < limit:
        for group in groups:
            if offset < len(group) and len(result) < limit:
                result.append(group[offset])
        offset += 1
    return sorted(result)


def load_checkpoint(path, device="cpu"):
    """只加载自己训练的 checkpoint；新版本明确使用 weights_only。"""
    kwargs = {"map_location": device}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = True
    return torch.load(path, **kwargs)


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
