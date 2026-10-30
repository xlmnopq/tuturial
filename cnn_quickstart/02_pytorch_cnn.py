"""
02_pytorch_cnn.py —— 用最小、可完整看清的代码，把"一个 CNN 从建到训"走一遍。

默认数据：Fashion-MNIST（服装 10 分类），图片 28×28 灰度，CPU 上 1 个 epoch 约 20~40 秒。
快速试跑（60 秒内出结果，准确率不会高，只验证流程能跑通）：
    python 02_pytorch_cnn.py --subset 6000 --epochs 1
正式训练（CPU 约 5~10 分钟，准确率约 90% 左右）：
    python 02_pytorch_cnn.py --epochs 10
在 d2l_pytorch_labs 的 GPU 环境里跑：
    python 02_pytorch_cnn.py --device cuda

新手请配合 03_NOTES.md 阅读；本文件按训练流程顺序组织：
    ① 导入与配置  ② 数据  ③ 模型  ④ 损失与优化器  ⑤ 训练循环  ⑥ 评估  ⑦ 保存/加载  ⑧ 预测与可视化
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent
CLASSES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


# --------------------------------------------------------------------------- #
# ① 配置：随机种子与设备
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    """固定随机种子，让"每次重跑结果基本一致"，方便排查问题。"""
    random.seed(seed)
    torch.manual_seed(seed)


def choose_device(name: str) -> torch.device:
    """设备选择：'auto' 时有 GPU 用 GPU，没有就用 CPU。"""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        print("!! 当前 PyTorch 检测不到 CUDA，自动改用 CPU。")
        name = "cpu"
    return torch.device(name)


# --------------------------------------------------------------------------- #
# ② 数据：Dataset（一条条样本）→ DataLoader（一批批样本）
# --------------------------------------------------------------------------- #
def make_loaders(batch_size: int, subset: int, data_dir: Path, seed: int):
    # 训练集要"抖动一下"：随机水平翻转 + 随机剪裁，这叫数据增广(data augmentation)，
    # 让模型见到的图片更多样、更不容易死记硬背。测试集绝不能增广，只用同样的大小和归一化。
    train_tf = transforms.Compose([
        transforms.RandomCrop(28, padding=4),      # 上下左右补 4 像素再随机裁回 28×28
        transforms.RandomHorizontalFlip(),          # 50% 概率左右翻转
        transforms.ToTensor(),                      # PIL 图片 → 张量，形状 (1,28,28)，取值缩到 [0,1]
        transforms.Normalize((0.2860,), (0.3530,)),  # 用均值/标准差把像素标准化到 0 附近
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])

    # datasets.FashionMNIST 会自动下载并缓存到 data_dir；两个 Dataset 指向同一份文件、不同 transform。
    train_ds = datasets.FashionMNIST(str(data_dir), train=True, download=True, transform=train_tf)
    test_ds = datasets.FashionMNIST(str(data_dir), train=False, download=True, transform=test_tf)

    if subset > 0:  # 只取一小部分做快速试跑
        generator = torch.Generator().manual_seed(seed)
        ids = torch.randperm(len(train_ds), generator=generator)[:subset].tolist()
        train_ds = Subset(train_ds, sorted(ids))
        test_ds = Subset(test_ds, sorted(torch.randperm(len(test_ds))[: max(subset // 5, 1000)].tolist()))

    # DataLoader：按 batch 打包样本，shuffle=True 表示每个 epoch 打乱顺序（训练集才需要）。
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


# --------------------------------------------------------------------------- #
# ③ 模型
# --------------------------------------------------------------------------- #
class SmallCNN(nn.Module):
    """一个足够小、能完整看懂的 CNN。

    输入 (N, 1, 28, 28)   N = batch size
      Conv2d(1,32,3,padding=1) + ReLU  → (N, 32, 28, 28)   每个输出通道是一种"局部模式检测器"
      MaxPool2d(2)                     → (N, 32, 14, 14)   空间尺寸减半，保留最强的响应
      Conv2d(32,64,3,padding=1) + ReLU → (N, 64, 14, 14)   在更抽象的特征上继续找模式
      MaxPool2d(2)                     → (N, 64, 7, 7)
      flatten(1)                       → (N, 64*7*7=3136)  把特征图拉平成向量
      Linear(3136, 128) + ReLU + Dropout → (N, 128)
      Linear(128, 10)                  → (N, 10)  logits（未归一化的分数）
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# 自定义模型时，先把形状逐层打印出来，比盯着报错信息猜更快：
#   print(SmallCNN()(torch.randn(2, 1, 28, 28)).shape)   # 期望 torch.Size([2, 10])


# --------------------------------------------------------------------------- #
# ④ 损失函数与优化器
# --------------------------------------------------------------------------- #
# nn.CrossEntropyLoss 内部 = log_softmax + NLLLoss，所以模型直接输出 logits，不要再手动 softmax！
criterion = nn.CrossEntropyLoss()


def make_optimizer(model, lr):
    # SGD + momentum 是经典搭配；Adam 通常更容易调，但小数据集上差别不大。
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)


# --------------------------------------------------------------------------- #
# ⑤ 训练一个 epoch：深度学习的 5 个标准动作
# --------------------------------------------------------------------------- #
def train_one_epoch(model, loader, optimizer, device, epoch, total_epochs):
    model.train()  # 训练模式：Dropout 会生效，BatchNorm 会更新统计量
    running_loss, correct, seen = 0.0, 0, 0
    for step, (images, labels) in enumerate(loader, 1):
        images, labels = images.to(device), labels.to(device)  # CPU 数据 → 设备

        optimizer.zero_grad()          # 1) 清空上一步的梯度
        logits = model(images)         # 2) 前向传播：算预测
        loss = criterion(logits, labels)  # 3) 算损失：预测与真实标签差多少
        loss.backward()                # 4) 反向传播：自动求出每个参数的梯度
        optimizer.step()               # 5) 参数更新：参数 -= 学习率 × 梯度

        running_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        seen += labels.size(0)
        if step % 200 == 0:
            print(f"  epoch {epoch}/{total_epochs}  step {step}/{len(loader)}  "
                  f"loss={running_loss / seen:.4f}  acc={correct / seen:.3f}")
    return running_loss / seen, correct / seen


# --------------------------------------------------------------------------- #
# ⑥ 评估：只前向传播，不算梯度
# --------------------------------------------------------------------------- #
@torch.no_grad()  # 等价于 with torch.no_grad(): 省显存、加速
def evaluate(model, loader, device):
    model.eval()  # 评估模式：关闭 Dropout，BatchNorm 用固定统计量（忘了写会导致结果异常）
    correct, seen = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        correct += (model(images).argmax(1) == labels).sum().item()
        seen += labels.size(0)
    return correct / seen


def top_confusions(model, loader, device, classes=CLASSES, k=3):
    """找出最容易混淆的 k 个类别对，比只看总准确率有信息量。"""
    model.eval()
    n = len(classes)
    # 混淆矩阵：行 = 真实类别，列 = 预测类别，对角线是预测对的张数
    matrix = torch.zeros(n, n, dtype=torch.long)
    with torch.no_grad():
        for images, labels in loader:
            pred = model(images.to(device)).argmax(1).cpu()
            # 把 (真实, 预测) 压成一个 0~n²-1 的编号，用 bincount 一次统计所有格子
            matrix += torch.bincount(labels * n + pred, minlength=n * n).reshape(n, n)
    pairs, off = [], matrix.clone()
    off.fill_diagonal_(0)  # 对角线是"预测正确"，找混淆要看非对角线
    for _ in range(k):
        i, j = divmod(int(off.argmax()), n)
        pairs.append((classes[i], classes[j], int(off[i, j])))
        off[i, j] = 0
    return matrix, pairs


# --------------------------------------------------------------------------- #
# ⑦ 保存与加载
# --------------------------------------------------------------------------- #
def save_checkpoint(model, path):
    # 只保存 state_dict（参数字典），不保存整个模型对象，加载时更安全、更可移植。
    torch.save({"model_state": model.state_dict(), "classes": CLASSES}, path)


def load_checkpoint(model, path, device):
    # weights_only=True 是 PyTorch 2.6+ 的默认值，能防止加载来源不明的代码。
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    return ckpt


# --------------------------------------------------------------------------- #
# ⑧ 预测：单张图推理
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict(model, image_tensor, device):
    model.eval()
    logits = model(image_tensor.unsqueeze(0).to(device))  # 单张图要加 batch 维 → (1,1,28,28)
    probs = logits.softmax(dim=1)[0]                     # 训练时不做 softmax，预测时才做，用来解释置信度
    top = int(probs.argmax())
    return CLASSES[top], float(probs[top]), probs.cpu()


def visualize_predictions(model, loader, device, path):
    """把一批测试图的预测结果画出来，肉眼看看模型错在哪。"""
    try:
        import matplotlib
        matplotlib.use("Agg")  # 不弹窗，直接存成 png（服务器/沙箱环境更稳妥）
        import matplotlib.pyplot as plt
    except ImportError:
        print("    （没装 matplotlib，跳过可视化）")
        return
    images, labels = next(iter(loader))
    model.eval()
    with torch.no_grad():
        preds = model(images.to(device)).argmax(1).cpu()
    fig, axes = plt.subplots(3, 6, figsize=(12, 6))
    for ax, image, label, pred in zip(axes.ravel(), images, labels, preds):
        ax.imshow(image.squeeze().numpy() * 0.3530 + 0.2860, cmap="gray")  # 反标准化，恢复肉眼可见亮度
        ax.axis("off")
        color = "green" if label == pred else "red"
        ax.set_title(f"真实: {CLASSES[label]}\n预测: {CLASSES[pred]}", fontsize=8, color=color)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print("  已保存预测可视化:", path)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="最小可训练 CNN：Fashion-MNIST 10 分类")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", help="auto / cpu / cuda")
    p.add_argument("--subset", type=int, default=0, help="只取多少个训练样本做快速试跑，0 表示全部")
    p.add_argument("--data-dir", default=str(ROOT / "data"))
    p.add_argument("--run-dir", default=str(ROOT / "runs" / "fashion_cnn"), help="结果目录，默认重新运行时覆盖上次的结果")
    args = p.parse_args()

    set_seed(args.seed)
    device = choose_device(args.device)
    print(f"设备: {device}   PyTorch: {torch.__version__}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_loader, test_loader = make_loaders(args.batch_size, args.subset, Path(args.data_dir), args.seed)
    print(f"训练样本: {len(train_loader.dataset)}   测试样本: {len(test_loader.dataset)}   "
          f"batch 数/轮: {len(train_loader)}")

    model = SmallCNN().to(device)
    n_params = sum(x.numel() for x in model.parameters())
    print(f"模型参数量: {n_params:,}")

    optimizer = make_optimizer(model, args.lr)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    best_acc, started = 0.0, time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, epoch, args.epochs)
        test_acc = evaluate(model, test_loader, device)
        marker = ""
        if test_acc > best_acc:  # 用验证/测试表现挑最佳权重（真实项目要留出独立的验证集）
            best_acc, marker = test_acc, "  ← 目前最好，已保存"
            save_checkpoint(model, run_dir / "best.pt")
        print(f"[epoch {epoch}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"test_acc={test_acc:.4f}{marker}")

    print(f"\n训练完成，用时 {time.time() - started:.1f}s，最佳 test acc = {best_acc:.4f}")

    # 从磁盘重新加载最佳权重，演示"部署时"的用法
    load_checkpoint(model, run_dir / "best.pt", device)
    matrix, pairs = top_confusions(model, test_loader, device)
    print("最易混淆的类别对（真实 -> 被预测成）:")
    for real, wrong, count in pairs:
        print(f"   {real} → {wrong}: {count} 张")

    name, prob, _ = predict(model, test_loader.dataset[0][0], device)
    print(f"单张预测示例: {name}  (置信度 {prob:.3f})")
    visualize_predictions(model, test_loader, device, run_dir / "predictions.png")
    torch.save(matrix, run_dir / "confusion_matrix.pt")  # 10×10 混淆矩阵，可自行 print(matrix)


if __name__ == "__main__":
    main()