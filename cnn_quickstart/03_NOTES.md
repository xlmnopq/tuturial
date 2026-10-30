# 代码解读笔记：从「看懂」到「改得动」

本笔记对应目录 `D:\program\tuturial\tuturial\d2l_pytorch_labs`（下称 **labs 项目**）里的真实课程代码。
目标是：看完这一页，你能指着任何一行说清楚它在 CNN 里负责什么、想调参时改哪一行。

---

## 0. 先建立整体地图

任何 CNN 项目（包括 labs 项目）都可以拆成 **6 个部件**，读代码时按这个顺序找就永远不会迷路：

| 部件 | 作用 | 一句话理解 | labs 项目中的位置 |
|---|---|---|---|
| 数据 | 把文件变成 `(图像, 标签)` 批次 | 数据是原材料 | `labkit/data.py` |
| 模型 | 定义"特征提取 + 分类"的计算图 | 参数在这里 | `labkit/image_models.py` |
| 损失 | 用一个数字衡量"错得有多离谱" | 优化的目标 | `labkit/engine.py` 第 37 行 |
| 优化器 | 根据梯度更新参数 | 怎么往山下走 | `train_image.py` 第 47 行 |
| 训练循环 | 反复：前向→算损失→反传→更新 | 主循环 | `labkit/engine.py` 的 `fit` / `run_epoch` |
| 评估与保存 | 选最佳模型、测试、出图表 | 检验成果 | `labkit/engine.py`、`labkit/plots.py` |

对应到模型和训练的**最小骨架**（背诵级别）：

```python
model = MyCNN().to(device)
loss_fn = nn.CrossEntropyLoss()                 # 分类任务固定用它
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for images, labels in train_loader:             # ① 取一个 batch
    images, labels = images.to(device), labels.to(device)
    optimizer.zero_grad()                       # ② 清梯度
    logits = model(images)                      # ③ 前向
    loss = loss_fn(logits, labels)              # ④ 算损失
    loss.backward()                             # ⑤ 反传
    optimizer.step()                            # ⑥ 更新
```

`engine.run_epoch`（training=True 分支）只是在上面 6 步外面套了"统计准确率、记录混淆矩阵、日志打印"等工程代码。

---

## 1. 模型读懂：`labkit/image_models.py`

### 1.1 读一个 `nn.Module`，只看两个方法

```python
class SmallCNN(nn.Module):        # ① 所有模型都继承 nn.Module
    def __init__(self):
        super().__init__()        # ② 必须先调用父类初始化
        self.features = nn.Sequential(...)   # ③ 含参数/含状态的层要放进 self.xxx
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):         # ④ 定义"输入怎么变成输出"
        return self.classifier(self.features(x).flatten(1))
```

要点：
- **`__init__` 里"申明"零件，`forward` 里"连线"**。PyTorch 只认 `self.xxx = ...` 这种赋值来登记参数，所以层千万别写成局部变量 `conv = nn.Conv2d(...)`（它照样能算，但优化器看不到它、也不会被保存）。
- **永远不要直接调用 `model.forward(x)`，要写 `model(x)`**。`model(x)` 除了 `forward` 还会触发 hook 等机制，这是官方约定。
- `nn.Sequential` = 按顺序执行，适合"一条直线"的结构。想写跳跃连接（ResNet）就必须在 `forward` 里手写 `out = out + shortcut`。

### 1.2 每一层在做什么（以 SmallCNN 为例）

```
输入 (N, 3, 32, 32)
Conv2d(3,32,3,padding=1) + ReLU   → (N, 32, 32, 32)   ← "空间不变"的局部特征提取
MaxPool2d(2)                      → (N, 32, 16, 16)   ← 空间减半
Conv2d(32,64,3,padding=1) + ReLU  → (N, 64, 16, 16)
MaxPool2d(2)                      → (N, 64, 8, 8)
Conv2d(64,128,3,padding=1) + ReLU → (N, 128, 8, 8)
AdaptiveAvgPool2d(1)              → (N, 128, 1, 1)    ← 自适应池化：不管输入多大都压成 1×1
flatten(1)                        → (N, 128)
Linear(128, 10)                   → (N, 10)           ← logits（分类分数，未归一化）
```

**CNN 的设计口诀**：越往后，空间尺寸越小（H×W 持续收缩），通道数越多（C 持续增大），最后压缩成一个向量做分类。

**尺寸公式**：`out = floor((in + 2*padding - kernel) / stride) + 1`
- `3×3` 卷积核 + `padding=1` + `stride=1` → 尺寸不变（现代 CNN 的默认写法）
- `MaxPool2d(2)` → 尺寸减半
- 用 `AdaptiveAvgPool2d(1)` 替代"大的全连接层"，可以让模型适应任意输入尺寸，也大幅减少参数量（VGG 原版 4096 维全连接层有几千万参数）。

### 1.3 为什么需要 ReLU、池化、Dropout、BatchNorm

| 组件 | 作用 | 少了会怎样 |
|---|---|---|
| ReLU | 引入**非线性** | 多层卷积叠起来依然等价于一个线性变换，网络再深也没用 |
| MaxPool / stride>1 | **降采样**：扩大感受野、减少计算量 | 全尺寸计算又慢又容易过拟合 |
| Dropout（如 AlexNet 里的 0.5） | 训练时随机丢弃神经元，防过拟合 | 小数据集上容易"背题"，训练准测试差 |
| BatchNorm | 让每层输出分布稳定，训练更快更稳 | 需要更小心地调学习率和初始化 |
| `AdaptiveAvgPool2d(1)` | 把每通道压成平均值 | 需要写死输入尺寸 |

### 1.4 读代码时值得注意的两个写法

- **`vgg_block(n_convs, in_channels, out_channels)`（第 45–53 行）**：用一个函数按"配置表"批量生成卷积块，`nn.Sequential(*layers)` 里的 `*` 是解包。这是写"同构层堆叠"的标准套路：VGG 的 `1,1,2,2,2` 就是循环里的一组参数。
- **`build_image_model(...)`（第 73 行起）**：工厂函数，用 `name` 选择结构。`resnet18` 分支里有三行最值得记住——**迁移学习的三种改法**：

```python
net = models.resnet18(weights=None)              # 随机初始化
net.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)  # 换输入层适配 32×32
net.maxpool = nn.Identity()                     # 去掉一次下采样
net.fc = nn.Linear(net.fc.in_features, 10)      # 换分类头适配 10 类
```

常见坑：**改了 `net.fc`，参数就被"换新"了**，新层是随机初始化的，所以迁移学习要先冻结骨干再替换头（见 `freeze_backbone` 分支）。AlexNet / VGG 的最后一层叫 `classifier[-1]`，不是 `fc`——这就是为什么要用 `net.classifier[-1] = nn.Linear(...)`。

---

## 2. 数据读懂：`labkit/data.py`

### 2.1 三段式：Dataset → transform → DataLoader

```python
train_source = datasets.CIFAR10(root, train=True, download=True, transform=train_tf)
val_source   = datasets.CIFAR10(root, train=True, download=False, transform=eval_tf)  # 同一份文件、不同 transform
subset       = Subset(train_source, train_ids)      # 只取划分到的索引
loader       = DataLoader(subset, batch_size=64, shuffle=True, num_workers=0)
for images, labels in loader: ...                    # images: (64, 3, 32, 32)  labels: (64,)
```

- **`Dataset`**：实现 `__len__` 和 `__getitem__`，返回一条样本（`(张量, 标签)`）。
- **`transform`**：在 `__getitem__` 里对单张图做处理。注意**训练用增广版、验证/测试用不增广版**，代码里用两个指向同一份数据的 Dataset 实现，别写混。
- **`DataLoader`**：负责组 batch、打乱、多进程读取。`shuffle=True` 只在训练集上开——验证/测试集乱序没意义，还会让结果难复现。
- **`num_workers`**：Windows 或 Jupyter 里默认写 0（主进程读数据），否则容易出现多进程启动问题；命令行脚本在 `if __name__ == "__main__":` 保护下可以设 4。

### 2.2 为什么 `ToTensor()` 和 `Normalize` 是必需的

- `ToTensor()`：把 PIL 图片通道从 HWC 变为 **CHW**（PyTorch 约定），并且把像素从 `[0,255]` 缩放到 `[0,1]`。
- `Normalize(mean, std)`：把每个通道标准化成均值 0、标准差 1。网络对这种尺度更友好，收敛更快。

### 2.3 训练/验证/测试为什么要分开

| 集合 | 用途 | labs 项目的做法 |
|---|---|---|
| 训练集 | 拿来算梯度和更新参数 | 50000 张里的 90%（45000） |
| 验证集 | 选"最佳 epoch"、调超参 | 50000 张里的 10%（5000） |
| 测试集 | **只用来报告最终成绩** | 官方 10000 张 |

`labkit/common.py` 的 `fixed_split` 用固定种子+按类别分层把训练集切成 train/val，并缓存成 `data/splits/*.json`。
如果把测试集用于选模型，报告出来的分数就会偏高，这叫做"数据泄漏"，是初学者最容易犯的评估错误。

---

## 3. 训练读懂：`labkit/engine.py`

### 3.1 `run_epoch`：一个函数同时处理训练和评估

关键理解：**训练和评估共用"前向"逻辑，区别只在四件事**。

| | 训练 | 评估 |
|---|---|---|
| `model.train()` / `model.eval()` | `train()` | `eval()` |
| 是否算梯度 | `torch.set_grad_enabled(True)` | 关闭（代码用 `optimizer is None` 判断） |
| 是否反传 + 更新 | 是 | 否 |
| 指标 | loss + acc | loss + acc + macro-F1 + 混淆矩阵 |

代码里"没有优化器 = 验证/测试"这个约定（第 32 行）非常值得学：它避免了为评估另写一份重复的前向代码。

### 3.2 训练循环外的三个"容易被忽略但很关键"的细节

1. **`optimizer.zero_grad(set_to_none=True)`**（第 44 行）：PyTorch 的梯度是**累加**的，不清零就会把上一步梯度也加进来。
2. **`nn.CrossEntropyLoss` 已经包含 softmax**（第 37 行）：模型要输出 **logits**，不要自己加 `softmax`。如果你在模型末尾加了 softmax，再配 CrossEntropyLoss，就等于 softmax 做了两次，梯度会很小、训练变慢。
3. **`torch.isfinite(loss)` 检查**（第 50 行）：一旦出现 NaN/Inf 直接报错，比训练完才发现"准确率永远是 10%"要好。NaN 的常见原因：学习率太大、损失里对 `log(0)` 取对数、除以 0。

### 3.3 `fit`：为什么保存的是"最佳"参数

```python
if val["accuracy"] > best_acc:                                   # 按验证集选 epoch
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}  # 拷到 CPU
    torch.save({"model_state": state, "config": config, ...}, run_dir / "best.pt")
...
model.load_state_dict(load_checkpoint(run_dir / "best.pt")["model_state"])  # 训练后恢复最佳
```

三点经验：
- **用验证集而不是训练集来选模型**。训练集上的 loss 总在下降，但训练 loss 下降不代表变好（过拟合时反而更差）。
- **`state_dict` 放到 CPU 再存**，这样在有 GPU 的机器上训、在没 GPU 的机器上推理也能加载。
- **训练后要 `load_state_dict` 回最佳权重**，否则测试的是"最后一个 epoch"而不是"最好的 epoch"。

---

## 4. 常见坑清单（新手翻车排行榜）

| # | 现象 | 原因 | 修复 |
|---|---|---|---|
| 1 | 准确率一直 ≈ 1/类别数（如 10%） | 忘了 `zero_grad` / 忘了 `optimizer.step()` / 标签与 logits 顺序搞反 | 对照"6 步骨架"逐行检查 |
| 2 | loss 变成 `nan` | 学习率太大 | 降到 1e-3 或 1e-4 再试 |
| 3 | 算出结果但准确率极低 | 忘了 `model.eval()`，Dropout 还在随机丢神经元 | 评估前 `model.eval()` |
| 4 | 报错信息里有 "size mismatch" | 输入通道数或形状不对 | 写个 `print(x.shape)` 逐层检查 |
| 5 | 报错 "Expected 4-dimensional input" | 忘了 batch 维度，单张图要 `x.unsqueeze(0)` | 补上 `N` 维 |
| 6 | 训练集 99%、测试集 65% | 过拟合 | 用更多数据/增广/Dropout/权重衰减；或减少参数量 |
| 7 | `torch.cuda.is_available()` 是 False | 装成了 CPU 版 PyTorch | 按官方安装页装对应 CUDA 版本的 torch |
| 8 | 想改模型却报 "Cannot assign ..." | 在 `__init__` 之外赋值了层，或没有 `super().__init__()` | 层的申明放到 `__init__` |
| 9 | 改了 `net.fc` 之后准确率反而下降 | 新分类头是**随机初始化**的 | 先冻结骨干只训分类头，再低学习率微调（见 labs 的 `--freeze-backbone`） |
| 10 | DataLoader 在 Windows 上卡死/反复执行 | 多进程启动问题 | `num_workers=0`，训练入口放进 `if __name__ == "__main__":` |

**调试方法（比读文档有用）**：先让模型在极小数据上**过拟合**（例如 200 张图训练 100 轮，准确率应该接近 100%）。如果做不到，说明数据、标签、损失或模型逻辑有 bug，而不是"数据不够"。

---

## 5. 调参指南：先改哪几个

按"性价比"排序：

1. **学习率 `lr`**：最敏感的超参数。SGD 常用 1e-2 量级，Adam 常用 1e-3。loss 抖动/NaN → 调小；loss 半天不动 → 调大 10 倍试试。
2. **epoch / 训练时长**：先看验证 loss 是否还在下降，没收敛就加轮数。
3. **batch size**：显存不够就减半；batch 变大时，学习率一般按比例适度调大。
4. **数据增广**：小数据集上最有效的"防过拟合"手段，比调网络结构的收益更直接。
5. **模型深度/宽度**：最后再动。先 baseline（SmallCNN）跑通，再换 AlexNet → VGG → ResNet，一次只改一个变量，记录结果。

labs 的 `compare_runs.py` 就是为"一次只改一个变量、横向对比"设计的：

```bash
python compare_runs.py --runs runs/smallcnn runs/alexnet runs/vgg11 runs/resnet18 --output runs/image_comparison.csv
```

---

## 6. 下一步深入了解的地方

- **切片、广播、`view` vs `reshape`**：`x.flatten(1)`、`x.permute()`、`x.contiguous()` 的区别，写 labkit 里的可视化代码（`plots.show_images`）时会天天遇到。
- **感受野**：为什么"3×3 的卷积堆两层"等效于"5×5 的卷积"，但参数更少、非线性更强（VGG 的核心论点）。
- **BatchNorm / 残差连接**：ResNet18 为什么能训练到几十层。labs 的 `notebooks/04_resnet18.ipynb` 可以对照读。
- **迁移学习**：`torchvision.models` + 冻结骨干 + 换分类头，labs README 的"扩展"一节有可直接运行的命令。