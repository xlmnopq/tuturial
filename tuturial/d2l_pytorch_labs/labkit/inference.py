"""从保存的配置和权重恢复模型，复用训练时的预处理。"""
import json
from pathlib import Path

import torch
from PIL import Image

from .common import choose_device, load_checkpoint
from .data import Vocabulary, collate_reviews, image_transforms, tokenize
from .image_models import build_image_model
from .text_models import build_text_model


def restore_run(run_dir, device="auto"):
    run_dir = Path(run_dir)
    device = choose_device(device)
    checkpoint = load_checkpoint(run_dir / "best.pt", "cpu")
    config = checkpoint["config"]
    manifest = json.loads((run_dir / "data_manifest.json").read_text(encoding="utf-8"))
    vocab = None
    if config["task"] == "image":
        # checkpoint 已包含全部权重，恢复时不再下载 ImageNet 权重。
        model = build_image_model(config["model"], config["variant"])
    else:
        vocab = Vocabulary(json.loads((run_dir / "vocab.json").read_text(encoding="utf-8")))
        model = build_text_model(config["model"], len(vocab))
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval(), config, manifest, vocab, device


@torch.no_grad()
def predict_image(run_dir, image_path, device="auto", top_k=3):
    model, config, manifest, _, device = restore_run(run_dir, device)
    if config["task"] != "image":
        raise ValueError("此运行目录不是图像分类模型")
    _, transform, _, _ = image_transforms(config["variant"], augment=False)
    image = Image.open(image_path).convert("RGB")
    if config["variant"] == "cifar":
        image = image.resize((32, 32))
    tensor = transform(image).unsqueeze(0).to(device)
    values, indices = model(tensor).softmax(1)[0].topk(min(top_k, len(manifest["classes"])))
    return [{"label": manifest["classes"][i], "probability": float(v)}
            for v, i in zip(values.cpu().tolist(), indices.cpu().tolist())]


@torch.no_grad()
def predict_text(run_dir, text, device="auto"):
    model, config, manifest, vocab, device = restore_run(run_dir, device)
    if config["task"] != "text":
        raise ValueError("此运行目录不是文本模型")
    if not tokenize(text):
        raise ValueError("请输入非空英文影评")
    ids = vocab.encode(text, config["max_length"])
    tokens, lengths, _ = collate_reviews([(torch.tensor(ids), 0)])
    probabilities = model(tokens.to(device), lengths.to(device)).softmax(1)[0].cpu()
    label = int(probabilities.argmax())
    return {"label": manifest["classes"][label],
            "probabilities": dict(zip(manifest["classes"], probabilities.tolist())),
            "used_tokens": len(ids), "unknown_fraction": ids.count(1) / len(ids)}
