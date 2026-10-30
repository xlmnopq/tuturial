# CNN 快速上手：从「看懂卷积」到「跑通训练」

> 适用对象：刚学完卷积神经网络（卷积、池化、ReLU、全连接、感受野等概念）的初学者。
> 本目录目标：用**最短的路径**把概念和代码对上号，并能在自己的电脑上真正跑起来。

本目录和你之前生成的 `d2l_pytorch_labs` 是**配套关系**、不是替代关系：

- `d2l_pytorch_labs` 是**完整课程项目**（9 个 notebook、4 种图像模型、数据划分、结果目录、对比脚本），适合做正式实验；
- `cnn_quickstart`（本目录）是**入门最小集**，只学"一个 CNN 从建到训"，每一行都读得懂，跑完再去读大项目就不会迷路。

---

## 一、目录预览

```
cnn_quickstart/
├── README.md            ← 你正在看的文件：学习路线 + 环境 + 全部命令
├── 01_conv_math.py      ← 零依赖，纯 Python 手写卷积/池化/尺寸公式（先跑这个）
├── 02_pytorch_cnn.py    ← 最小可训练 CNN：Fashion-MNIST 10 分类（约 160 行，全部带注释）
└── 03_NOTES.md          ← d2l_pytorch_labs 代码逐段解读 + 踩坑清单 + 调参指南
```

---

## 二、当前电脑环境（我先帮你检查过了）

| 项目 | 现状 | 结论 |
|---|---|---|
| GPU | NVIDIA GTX 1060 3GB，驱动 560.94（支持 CUDA 12.6） | 能用 GPU，但显存小：batch size ≤ 64，别跑大模型 |
| Python（唯一可用的） | `D:\APPstore\anaconda3\python.exe`（Python 3.14.6） | **可以使用**，但**没装 PyTorch** |
| 其它 Python | `C:\Users\xlmn\AppData\Local\Microsoft\WindowsApps\python.exe` 只是微软商店的占位程序，运行会直接报错退出 | 不要用它 |
| d2l_pytorch_labs 里的 `runs/` | 有 1 个 epoch 的 smoke 测试结果（torch 1.12.1+cu113 环境） | 说明那个环境不在当前电脑，需要重装 |
| 网络 | 我这边测试 pypi.org 和清华镜像都连不上 | 下面安装命令请你在自己的终端运行 |

---

## 三、先跑不装任何东西的两个文件

`01_conv_math.py` **只用标准库**，任何 Python 都能直接运行：

```powershell
cd D:\program\tuturial\cnn_quickstart
& D:\APPstore\anaconda3\python.exe 01_conv_math.py
```

它会打印 5 个演示，建议对照输出理解：

1. **边缘检测**：一个 3×3 的 Sobel 卷积核在 6×6 的图上滑动，只在"0 和 1 交界"的那一列输出大值 → 卷积核 = 模式检测器。
2. **三个卷积核 → 三个特征图**：`C_out` 就是"卷积核的个数"。
3. **多通道卷积**：`nn.Conv2d(3, 2, 3)` 的权重形状是 `(2, 3, 3, 3)`——每个输出通道的卷积核在所有输入通道上分别卷，再相加。
4. **尺寸公式**：`out = floor((in + 2p - k)/s) + 1`，以及 `pool=2` 就是 PyTorch 里的 `MaxPool2d(2)`。
5. **完整走一遍 SmallCNN**：看清 `(3,32,32) → (32,32,32) → (32,16,16) → … → (10,)` 每一步的形状。

`03_NOTES.md` 是纯文本笔记，也可以先通读一遍第 0 节的"6 大部件"表格。

---

## 四、安装 PyTorch（只装一次）

`02_pytorch_cnn.py` 需要 `torch` 和 `torchvision`。你的显卡是 GTX 1060（Pascal 架构），**任何 CUDA 版本都能用**，选新版即可。

> ⚠️ 建议**不要**把 PyTorch 装进 anaconda 的 base 环境（base 被弄坏后很难修）。推荐用 conda 新建一个环境：

```powershell
# 1) 新建环境（Python 3.11，兼容性最好）
D:\APPstore\anaconda3\Scripts\conda.exe create -n cnn python=3.11 -y

# 2) 在该环境里装 GPU 版 PyTorch（CUDA 12.6 轮子，GTX 1060 可用）
D:\APPstore\anaconda3\envs\cnn\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 国内网络慢时，换成镜像源（两行里用一行即可）：
# D:\APPstore\anaconda3\envs\cnn\python.exe -m pip install torch torchvision --index-url https://mirrors.aliyun.com/pytorch-wheels/cu126

# 3) 可视化要用的 matplotlib
D:\APPstore\anaconda3\envs\cnn\python.exe -m pip install matplotlib

# 4) 验证：应打印 True 和显卡名
D:\APPstore\anaconda3\envs\cnn\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

想偷懒、不新建环境？直接把上面命令里的 `D:\APPstore\anaconda3\envs\cnn\python.exe` 换成 `D:\APPstore\anaconda3\python.exe` 即可，但会污染 base 环境（**不推荐**）。

> 如果装完 `torch.cuda.is_available()` 还是 False：先确认你用的就是这个环境的 python；再检查是否误装了 CPU 版；`nvidia-smi` 能看到显卡但 PyTorch 看不到，通常是版本不匹配，按 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/) 重新安装。

---

## 五、跑通第一个可训练的 CNN

```powershell
cd D:\program\tuturial\cnn_quickstart

# ① 60 秒快速试跑：只取 6000 张、1 个 epoch，只为确认"能跑通"
& D:\APPstore\anaconda3\envs\cnn\python.exe 02_pytorch_cnn.py --subset 6000 --epochs 1

# ② 正式一点：全量数据 3 轮（第一次会自动下载 Fashion-MNIST，约 30MB）
& D:\APPstore\anaconda3\envs\cnn\python.exe 02_pytorch_cnn.py --epochs 3

# ③ 有 GPU 就用 GPU（显卡只有 3GB，batch 用 64 更稳）
& D:\APPstore\anaconda3\envs\cnn\python.exe 02_pytorch_cnn.py --epochs 5 --batch-size 64 --device cuda
```

预期结果：全量数据、CPU 上 3 轮大约 90% 上下；GPU 上 5 轮通常 91~92%。
**这几个数字只是参考**，你的机器上略有差别很正常。

运行结束后在 `cnn_quickstart/runs/fashion_cnn/` 里会看到：

| 文件 | 内容 |
|---|---|
| `best.pt` | 测试准确率最高那一轮的参数（`state_dict`） |
| `predictions.png` | 18 张测试图的"真实 vs 预测"，绿色正确、红色错误 |
| `confusion_matrix.pt` | 10×10 混淆矩阵张量 |

同时终端会打印**最易混淆的类别对**（例如 Trouser ↔ Dress、Shirt ↔ Coat），这比只看一个准确率更能说明问题。

---

## 六、`02_pytorch_cnn.py` 的骨架（背下来就够用）

文件是按训练流程顺序排的，函数名前面的数字就是阅读顺序：

```
① set_seed / choose_device        固定随机种子、选 CPU 还是 GPU
② make_loaders                    Dataset → transform → DataLoader
③ SmallCNN                        模型：两个「卷积+ReLU」块 + 池化 + 全连接
④ criterion / make_optimizer      CrossEntropyLoss + SGD(momentum=0.9)
⑤ train_one_epoch                 训练循环 6 步：zero_grad → forward → loss → backward → step
⑥ evaluate / top_confusions       评估：model.eval() + torch.no_grad()
⑦ save_checkpoint / load_checkpoint  只存 state_dict
⑧ predict / visualize_predictions 单张推理 + 可视化
```

**六步训练骨架**（几乎所有 CNN 代码都是它的变体，labs 项目的 `engine.run_epoch` 也是）：

```python
optimizer.zero_grad()              # 1 清空上一步梯度（PyTorch 梯度默认累加！）
logits = model(images)             # 2 前向
loss = criterion(logits, labels)   # 3 算损失（模型输出 logits，不要再手动 softmax）
loss.backward()                    # 4 反向传播，自动求梯度
optimizer.step()                   # 5 参数更新
# 训练用 model.train()，评估用 model.eval() + torch.no_grad()
```

新手最容易错的三点，先记住：
1. **`model.train()` / `model.eval()` 必须切换**——否则 Dropout、BatchNorm 行为不对。
2. **`CrossEntropyLoss` 自带 softmax**——模型最后输出 logits，概率只在推理时算。
3. **输入形状是 `(N, C, H, W)`**——单张图推理要 `x.unsqueeze(0)` 补上 `N`。

---

## 七、把它换成 CIFAR-10（练手第一题）

Fashion-MNIST 是 1 通道 28×28，CIFAR-10 是 3 通道 32×32，需要改的地方只有 4 处：

| 位置 | 改法 |
|---|---|
| `make_loaders` | `datasets.FashionMNIST(...)` → `datasets.CIFAR10(..., transform=train_tf)`；增广加 `RandomCrop(32, padding=4)` + `RandomHorizontalFlip()` |
| `Normalize` | `((0.2860,), (0.3530,))` → `((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))` |
| `SmallCNN.__init__` | 第一个 `Conv2d(1, 32, ...)` → `Conv2d(3, 32, ...)` |
| `CLASSES` | 换成 `["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]` |

改完如果准确率在 70~80%，说明你已经跑通了 CIFAR-10 级别的 CNN。**这正是 labs 项目 `train_image.py --model smallcnn` 做的事情**（它的 SmallCNN 多了个 `AdaptiveAvgPool2d(1)`，所以输入尺寸可以更自由）。

---

## 八、接下来进入 labs 项目（正式实验）

labs 项目需要 `data/` 目录里有数据集缓存（它不会自动下载；`prepare_data.py` 负责下载）。切换到同一个环境执行：

```powershell
cd D:\program\tuturial\tuturial\d2l_pytorch_labs
& D:\APPstore\anaconda3\envs\cnn\python.exe prepare_data.py --dataset all      # 下载并校验 CIFAR-10 / IMDB
& D:\APPstore\anaconda3\envs\cnn\python.exe check_environment.py               # 环境自检

# 先用 smoke 模式（128 张训练图、1 轮）确认流程通，再正式训练
& D:\APPstore\anaconda3\envs\cnn\python.exe train_image.py --model smallcnn --smoke --run-dir runs/smoke_smallcnn

# 正式训练（3GB 显存建议 batch 32，可加 --amp 省显存）
& D:\APPstore\anaconda3\envs\cnn\python.exe train_image.py --model resnet18 --epochs 20 --batch-size 32 --amp --run-dir runs/resnet18
```

> 注意：labs 的 `runs/` 下**已有** `smallcnn_cifar_*` 等目录，并且里面有 `best.pt`。它的规则是"运行目录已有 best.pt 就拒绝启动"，所以要用**新的 `--run-dir` 名字**（上面的命令已经这样做了）。

### 代码阅读顺序建议

```
labkit/image_models.py  →  看懂 SmallCNN/AlexNet/VGG 的 forward
labkit/data.py          →  看懂 transform 与 DataLoader（02 里的写法是它的简化版）
labkit/engine.py        →  看懂 fit / run_epoch（02 里的训练循环是它的简化版）
train_image.py          →  参数解析 + 串起上面三块
notebooks/02_alexnet.ipynb → 04_resnet18.ipynb  按顺序跑实验
```

详细的"逐段解读"见 [03_NOTES.md](03_NOTES.md)：包括为什么 VGG 用 3×3 堆叠、Dropout 与 BatchNorm 的作用、迁移学习怎么改分类头、10 个新手坑、调参顺序。

---

## 九、学完这一圈，你应该能独立完成

- [ ] 不看书，手写出"训练一个批量输入的分类 CNN"的 6 步循环
- [ ] 给定任意输入尺寸和卷积配置，口算出每层输出尺寸
- [ ] 解释 `C_out`（卷积核个数）和 `C_in`（输入通道数）的区别，说清 `Conv2d(3,64,3)` 的权重形状
- [ ] 说清 `train()` / `eval()`、`no_grad()` 分别解决什么问题
- [ ] 遇到准确率停在 10%、loss 变 NaN，能按 [03_NOTES.md](03_NOTES.md) 第 4 节的清单定位问题
- [ ] 把 `02_pytorch_cnn.py` 改成 CIFAR-10 版本并训练到 70% 以上
- [ ] 在 labs 项目里跑通 AlexNet / VGG11 / ResNet18 各一次，并用 `compare_runs.py` 对比