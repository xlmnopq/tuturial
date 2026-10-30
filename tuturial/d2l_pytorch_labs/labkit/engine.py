"""训练、验证、最佳模型保存和最终测试。测试集不参与选模型。"""
from __future__ import annotations

import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn

from .common import load_checkpoint, save_json, write_csv


def unpack_batch(model, batch, device):
    if len(batch) == 2:
        images, labels = batch
        return model(images.to(device, non_blocking=True)), labels.to(device, non_blocking=True)
    tokens, lengths, labels = batch
    return model(tokens.to(device, non_blocking=True), lengths.to(device)), labels.to(device)


def make_scaler(enabled):
    # 同时兼容 PyTorch 2.x 和已有的 1.12 教学环境。
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def run_epoch(model, loader, device, optimizer=None, scaler=None, amp=False,
              grad_clip=1.0, freeze_backbone=False, log_every=100):
    """optimizer=None 是验证/测试；否则执行梯度清零、反传和参数更新。"""
    training = optimizer is not None
    model.train(training) # 训练模式下，BatchNorm 会累积统计量，Dropout 会随机"关闭"神经元
    if training and freeze_backbone:
        # 冻结骨干时，BatchNorm 统计量和 Dropout 也保持评估模式。
        model.eval() # 评估模式下，BatchNorm 会使用累积统计量，Dropout 会固定"关闭"神经元
    criterion = nn.CrossEntropyLoss() # 多分类损失函数，自动计算 softmax
    total_loss, correct, count = 0.0, 0, 0
    # 真实类别数从 logits 推断，兼容任意二分类/多分类模型。
    confusion = None
    started = time.perf_counter()
    for step, batch in enumerate(loader, 1):
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            context = torch.autocast(device_type="cuda", dtype=torch.float16) if amp and device.type == "cuda" else nullcontext()
            with context:
                logits, labels = unpack_batch(model, batch, device)
                loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("损失出现 NaN/Inf，请减小学习率或关闭 AMP")
            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
        predictions = logits.detach().argmax(dim=1)
        size = labels.size(0)
        total_loss += loss.item() * size
        correct += (predictions == labels).sum().item()
        count += size
        n_classes = logits.size(1)
        if confusion is None:
            confusion = torch.zeros(n_classes, n_classes, dtype=torch.long)
        flat = (labels.detach() * n_classes + predictions).cpu()
        confusion += torch.bincount(flat, minlength=n_classes**2).reshape(n_classes, n_classes)
        if training and log_every and step % log_every == 0:
            print("  batch %d/%d, loss=%.4f, acc=%.4f" %
                  (step, len(loader), total_loss / count, correct / count), flush=True)
    if not count:
        raise ValueError("DataLoader 没有样本")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    tp = confusion.diag().float()
    precision = tp / confusion.sum(0).clamp(min=1)
    recall = tp / confusion.sum(1).clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)
    seconds = time.perf_counter() - started
    return {"loss": total_loss / count, "accuracy": correct / count,
            "macro_f1": f1.mean().item(), "per_class_f1": f1.tolist(),
            "confusion_matrix": confusion.tolist(), "samples": count,
            "seconds": seconds, "samples_per_second": count / max(seconds, 1e-9)}


def fit(model, bundle, optimizer, device, run_dir, config, epochs=20,
        amp=False, grad_clip=1.0, freeze_backbone=False, scheduler=None):
    """只用验证准确率选择最佳 epoch。运行目录已有模型时要求另取名称。"""
    if epochs < 1:
        raise ValueError("epochs 至少为 1")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "best.pt").exists():
        raise FileExistsError("运行目录已有 best.pt，请使用新的 --run-dir，防止覆盖实验。")
    model.to(device)
    amp = bool(amp and device.type == "cuda")
    scaler = make_scaler(amp)
    save_json(run_dir / "config.json", config)
    save_json(run_dir / "data_manifest.json", bundle.meta)
    if bundle.vocab is not None:
        save_json(run_dir / "vocab.json", bundle.vocab.tokens)
    history, best_acc, best_epoch = [], -1.0, 0
    for epoch in range(1, epochs + 1):
        train = run_epoch(model, bundle.train, device, optimizer, scaler, amp,
                          grad_clip, freeze_backbone)
        val = run_epoch(model, bundle.val, device, amp=amp)
        row = {"epoch": epoch, "train_loss": train["loss"], "train_accuracy": train["accuracy"],
               "val_loss": val["loss"], "val_accuracy": val["accuracy"],
               "val_macro_f1": val["macro_f1"], "lr": optimizer.param_groups[0]["lr"],
               "train_seconds": train["seconds"], "val_seconds": val["seconds"]}
        history.append(row)
        write_csv(run_dir / "history.csv", history)
        print("Epoch %d/%d train_loss=%.4f train_acc=%.4f val_acc=%.4f" %
              (epoch, epochs, train["loss"], train["accuracy"], val["accuracy"]), flush=True)
        if val["accuracy"] > best_acc:
            best_acc, best_epoch = val["accuracy"], epoch
            # tensor 放在 CPU，使 checkpoint 可在有/无 GPU 环境中加载。
            state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            torch.save({"model_state": state, "config": config,
                        "best_epoch": best_epoch, "best_val_accuracy": best_acc}, run_dir / "best.pt")
        if scheduler is not None:
            scheduler.step()
    model.load_state_dict(load_checkpoint(run_dir / "best.pt", "cpu")["model_state"])
    return history


def final_test(model, bundle, device, run_dir):
    """训练完成且恢复最佳验证模型后，执行最终测试。"""
    metrics = run_epoch(model, bundle.test, device)
    metrics["classes"] = bundle.meta["classes"]
    metrics["test_scope"] = "full" if metrics["samples"] == (10000 if bundle.meta["dataset"] == "cifar10" else 25000) else "subset"
    save_json(Path(run_dir) / "test_metrics.json", metrics)
    print("Test [%s] n=%d accuracy=%.4f macro_f1=%.4f" %
          (metrics["test_scope"], metrics["samples"], metrics["accuracy"], metrics["macro_f1"]), flush=True)
    return metrics
