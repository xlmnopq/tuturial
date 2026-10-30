"""两个数据集的完整数据管线：下载、训练/验证划分、预处理、DataLoader。"""
from __future__ import annotations

import hashlib
import html
import json
import re
import tarfile
import urllib.request
from collections import Counter
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
import requests
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from .common import balanced_limit, fixed_split, save_json, seed_worker

IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
IMDB_SHA1 = "01ada507287d82875905620988597833ad4e0903"
CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog",
                 "frog", "horse", "ship", "truck"]
LABEL_NAMES = ["negative", "positive"]


@dataclass
class DataBundle:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    meta: dict
    vocab: object = None


def file_sha1(path):
    digest = hashlib.sha1()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_imdb(root, download=True, archive=None):
    """校验官方压缩包，只读取已标注影评，避免解压大量文件。"""
    folder = Path(root) / "imdb"
    folder.mkdir(parents=True, exist_ok=True)
    paths = [folder / "train.jsonl", folder / "test.jsonl"]
    if all(p.exists() for p in paths):
        return paths
    archive = Path(archive) if archive else folder / "aclImdb_v1.tar.gz"
    if not archive.exists():
        if not download:
            raise FileNotFoundError("IMDB 数据不存在。先运行 python prepare_data.py --dataset imdb")
        tmp = archive.with_suffix(archive.suffix + ".part")
        archive.parent.mkdir(parents=True, exist_ok=True)
        print("Downloading IMDB:", IMDB_URL, flush=True)
        with requests.get(IMDB_URL, stream=True, timeout=(20, 60)) as response, tmp.open("wb") as f:
            response.raise_for_status()
            total, last = 0, 0
            for chunk in response.iter_content(1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                total += len(chunk)
                if total - last >= 10 * 1024 * 1024:
                    print("Downloaded %.1f MB" % (total / 1024**2), flush=True)
                    last = total
        if file_sha1(tmp) != IMDB_SHA1:
            raise RuntimeError("IMDB 下载校验失败，请删除 .part 文件后重新下载。")
        tmp.replace(archive)
    if file_sha1(archive) != IMDB_SHA1:
        raise RuntimeError("IMDB 压缩包 SHA-1 不匹配，请使用官方原始数据包。")
    records = {"train": [], "test": []}
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            match = re.fullmatch(r"aclImdb/(train|test)/(neg|pos)/([^/]+\.txt)", member.name)
            if not match or not member.isfile():
                continue
            split, label, filename = match.groups()
            source = tar.extractfile(member)
            if source is None:
                continue
            text = source.read().decode("utf-8", errors="replace")
            records[split].append({"id": label + "/" + filename,
                                   "text": text, "label": int(label == "pos")})
    for split, path in zip(("train", "test"), paths):
        items = sorted(records[split], key=lambda x: x["id"])
        if len(items) != 25000 or sum(x["label"] for x in items) != 12500:
            raise RuntimeError("IMDB 标注样本数量异常")
        tmp = path.with_suffix(".part")
        with tmp.open("w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        tmp.replace(path)
    save_json(folder / "source.json", {"url": IMDB_URL, "sha1": IMDB_SHA1,
                                       "train": 25000, "test": 25000,
                                       "unlabeled_used": False})
    return paths


def image_transforms(variant="cifar", augment=True):
    if variant == "cifar":
        mean, std = (0.5,) * 3, (0.5,) * 3
        train_ops = ([transforms.RandomCrop(32, padding=4),
                      transforms.RandomHorizontalFlip()] if augment else [])
        eval_ops = []
    elif variant == "torchvision":
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        eval_ops = [transforms.Resize(256), transforms.CenterCrop(224)]
        train_ops = ([transforms.Resize(256), transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                      transforms.RandomHorizontalFlip()] if augment else list(eval_ops))
    else:
        raise ValueError("未知 variant")
    tail = [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(train_ops + tail), transforms.Compose(eval_ops + tail), mean, std


def make_loader(dataset, batch_size, shuffle, seed, workers=0, collate_fn=None):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, pin_memory=torch.cuda.is_available(),
                      worker_init_fn=seed_worker, generator=generator,
                      collate_fn=collate_fn, drop_last=False)


def load_image_data(root="data", batch_size=64, seed=42, workers=0,
                    variant="cifar", augment=True, download=True,
                    max_train=0, max_val=0, max_test=0):
    root = Path(root)
    train_tf, eval_tf, mean, std = image_transforms(variant, augment)
    # 两个 Dataset 指向同一份官方训练数据，但使用不同 transform。
    train_source = datasets.CIFAR10(str(root / "cifar10"), train=True,
                                    download=download, transform=train_tf)
    val_source = datasets.CIFAR10(str(root / "cifar10"), train=True,
                                  download=False, transform=eval_tf)
    test_source = datasets.CIFAR10(str(root / "cifar10"), train=False,
                                   download=False, transform=eval_tf)
    split_path = root / "splits" / ("cifar10_seed%d.json" % seed)
    train_ids, val_ids = fixed_split(train_source.targets, split_path, seed)
    train_ids = balanced_limit(train_ids, train_source.targets, max_train, seed)
    val_ids = balanced_limit(val_ids, train_source.targets, max_val, seed)
    test_ids = balanced_limit(range(len(test_source)), test_source.targets, max_test, seed)
    subsets = [Subset(train_source, train_ids), Subset(val_source, val_ids),
               Subset(test_source, test_ids)]
    loaders = [make_loader(d, batch_size, i == 0, seed, workers)
               for i, d in enumerate(subsets)]
    meta = {"dataset": "cifar10", "classes": CIFAR_CLASSES,
            "split_file": str(split_path), "seed": seed, "val_fraction": 0.1,
            "train_indices": train_ids, "val_indices": val_ids, "test_indices": test_ids,
            "sizes": dict(zip(["train", "val", "test"], map(len, subsets))),
            "variant": variant, "image_size": 32 if variant == "cifar" else 224,
            "augment": augment, "mean": list(mean), "std": list(std)}
    return DataBundle(*loaders, meta)


def tokenize(text):
    """针对 IMDB 英文影评的简单词级分词，不适用于中文分词。"""
    text = html.unescape(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    return re.findall(r"[a-z]+(?:'[a-z]+)?|\d+|[^\w\s]", text)


class Vocabulary:
    def __init__(self, tokens):
        self.tokens = list(tokens)
        if self.tokens[:2] != ["<pad>", "<unk>"] or len(set(self.tokens)) != len(self.tokens):
            raise ValueError("词表必须以 <pad>、<unk> 开头且没有重复词")
        self.stoi = {token: i for i, token in enumerate(self.tokens)}

    @classmethod
    def build(cls, texts, max_vocab=20000, min_freq=2):
        if max_vocab < 2 or min_freq < 1:
            raise ValueError("max_vocab >= 2 且 min_freq >= 1")
        counts = Counter(token for text in texts for token in tokenize(text))
        words = [word for word, count in sorted(counts.items(), key=lambda x: (-x[1], x[0]))
                 if count >= min_freq][:max_vocab - 2]
        return cls(["<pad>", "<unk>"] + words)

    def encode(self, text, max_length=256):
        if max_length < 5:
            raise ValueError("max_length 至少为 5，以兼容 TextCNN 最大卷积核")
        ids = [self.stoi.get(t, 1) for t in tokenize(text)[:max_length]]
        return ids or [1]

    def __len__(self):
        return len(self.tokens)


class ReviewDataset(Dataset):
    def __init__(self, records, vocab, max_length):
        self.records = list(records)
        self.ids = [vocab.encode(x["text"], max_length) for x in self.records]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return torch.tensor(self.ids[index], dtype=torch.long), self.records[index]["label"]


def collate_reviews(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(x) for x in sequences], dtype=torch.long)
    tokens = pad_sequence(sequences, batch_first=True, padding_value=0)
    # TextCNN 包含 3、4、5 三种卷积核。空文本也至少补齐到 5 个位置。
    if tokens.size(1) < 5:
        tokens = torch.nn.functional.pad(tokens, (0, 5 - tokens.size(1)))
    return tokens, lengths, torch.tensor(labels, dtype=torch.long)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_text_data(root="data", batch_size=64, seed=42, workers=0,
                   max_vocab=20000, min_freq=2, max_length=256, download=True,
                   max_train=0, max_val=0, max_test=0):
    root = Path(root)
    train_path, test_path = download_imdb(root, download)
    source_train, source_test = read_jsonl(train_path), read_jsonl(test_path)
    labels = [x["label"] for x in source_train]
    split_path = root / "splits" / ("imdb_seed%d.json" % seed)
    train_ids, val_ids = fixed_split(labels, split_path, seed)
    train_ids = balanced_limit(train_ids, labels, max_train, seed)
    val_ids = balanced_limit(val_ids, labels, max_val, seed)
    test_ids = balanced_limit(range(len(source_test)), [x["label"] for x in source_test], max_test, seed)
    train_records = [source_train[i] for i in train_ids]
    val_records = [source_train[i] for i in val_ids]
    test_records = [source_test[i] for i in test_ids]
    # 词表只拟合实际训练子集，验证和测试都不能进入词频统计。
    vocab = Vocabulary.build((x["text"] for x in train_records), max_vocab, min_freq)
    subsets = [ReviewDataset(records, vocab, max_length)
               for records in [train_records, val_records, test_records]]
    loaders = [make_loader(d, batch_size, i == 0, seed, workers, collate_reviews)
               for i, d in enumerate(subsets)]
    meta = {"dataset": "imdb", "classes": LABEL_NAMES,
            "split_file": str(split_path), "seed": seed, "val_fraction": 0.1,
            "train_indices": train_ids, "val_indices": val_ids, "test_indices": test_ids,
            "sizes": dict(zip(["train", "val", "test"], map(len, subsets))),
            "max_length": max_length, "max_vocab": max_vocab, "min_freq": min_freq,
            "vocab_size": len(vocab), "tokenizer_version": "english_regex_v1"}
    return DataBundle(*loaders, meta, vocab)
