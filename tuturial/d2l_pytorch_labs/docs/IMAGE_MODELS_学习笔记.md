# `image_models.py` 学习笔记

> 面向初学者的逐行精读笔记 + 实测数据。
> 所有参数量、每层输出形状都是**在这台机器上真实跑出来的**，不是估算。

| 项目 | 内容 |
|---|---|
| 被解读的文件 | [`labkit/image_models.py`](../labkit/image_models.py)（114 行） |
| 任务 | CIFAR-10 图像分类（10 类，32×32 彩色小图） |
| 环境 | torch 2.5.1+cu121 / torchvision 0.20.1+cu121 |
| 笔记性质 | 学习用，可反复查阅 |

---

## 目录

- [1. 一分钟速查表](#1-一分钟速查表)
- [2. 预备知识（必须先看）](#2-预备知识必须先看)
- [3. 文件整体结构](#3-文件整体结构)
- [4. `SmallCNN` — 最小可运行 CNN](#4-smallcnn--最小可运行-cnn)
- [5. `AlexNetCIFAR` — 第一个"深"网络](#5-alexnetcifar--第一个深网络)
- [6. `vgg_block` + `VGG11CIFAR` — 模块化积木](#6-vgg_block--vgg11cifar--模块化积木)
- [7. `ResNet18` — 残差网络](#7-resnet18--残差网络)
- [8. `build_image_model` — 模型工厂](#8-build_image_model--模型工厂)
- [9. `parameter_count` — 参数统计](#9-parameter_count--参数统计)
- [10. 实测数据汇总](#10-实测数据汇总)
- [11. 四模型横向对比与选型](#11-四模型横向对比与选型)
- [12. 常见坑位清单](#12-常见坑位清单)
- [13. 自测题](#13-自测题)
- [附录 A：一键验证脚本](#附录-a一键验证脚本)
- [附录 B：它在训练流程中的位置](#附录-b它在训练流程中的位置)
- [附录 C：术语表](#附录-c术语表)

---

## 1. 一分钟速查表

| 模型 | 参数量 | 卷积层数 | 特点 | CIFAR-10 经验准确率* |
|---|---:|---:|---|---|
| `SmallCNN` | **94,538** | 3 | 最简单，无 Dropout/BN | 65% ~ 75% |
| `AlexNetCIFAR` | **2,388,298** | 5 | ReLU + Dropout，2 层全连接 | 78% ~ 85% |
| `VGG11CIFAR` | **9,488,266** | 8 | 积木式堆叠，**输入尺寸锁死 32×32** | 82% ~ 88% |
| `ResNet18`(cifar) | **11,173,962** | 17 | 残差连接 + BatchNorm | 90% ~ 94% |

\* 经验区间，强依赖超参；**以你自己跑出来的 `history.csv` 为准**。

一句话记住每个模型：

- **SmallCNN**：卷积提特征 → 池化降维 → 全连接分类，CNN 的最小原型。
- **AlexNetCIFAR**：**ReLU + Dropout + 数据增强** 三件套的发明者。
- **VGG11CIFAR**：把"n 个 3×3 卷积 + 1 次池化"封装成积木，堆到 11 层。
- **ResNet18**：`out += identity` 给梯度修了一条高速公路。

---

## 2. 预备知识（必须先看）

### 2.1 张量形状约定：NCHW

PyTorch 里图像数据的形状永远是：

```
(batch_size, channels, height, width)
     N          C        H       W
```

CIFAR-10 的一张图：

```
输入图片: (1, 3, 32, 32)
           ↑  ↑  ↑   ↑
           │  │  │   └── 宽 32 像素
           │  │  └────── 高 32 像素
           │  └────────── 3 个通道（R、G、B）
           └────────────── 1 张图（batch 里的第 1 张）
```

> ⚠️ 这和 PIL / OpenCV 的 **HWC** 顺序（高、宽、通道）不同。`transforms.ToTensor()` 干的事之一就是把 HWC 转成 CHW。

### 2.2 卷积输出尺寸公式

```
H_out = floor( (H_in + 2×padding - kernel_size) / stride ) + 1
```

本项目所有卷积都是 `kernel=3, padding=1, stride=1`，代入：

```
H_out = (H_in + 2 - 3) / 1 + 1 = H_in
```

**结论：`3×3 + padding=1 + stride=1` 保持空间尺寸不变。** 这是"same padding"，全项目都在用。

### 2.3 池化输出尺寸

`MaxPool2d(2)`：`H_out = H_in / 2`

```
32 → 16 → 8 → 4 → 2 → 1
```

### 2.4 参数量公式（自己就能手算）

| 层类型 | 参数量 |
|---|---|
| 卷积 `Conv2d(C_in, C_out, k)` | `(k × k × C_in + 1) × C_out` |
| 卷积 **带 `bias=False`** | `k × k × C_in × C_out` |
| 批归一化 `BatchNorm2d(C)` | `2 × C`（γ 和 β） |
| 全连接 `Linear(in, out)` | `(in + 1) × out` |
| 池化 / ReLU / Flatten | **0** |

### 2.5 CNN 的两个通用设计模式

**模式一：空间变小，通道变多**

```
空间尺寸:  32 → 16 → 8 → 4 → 1      （越来越小）
通道数:     3 →  64 → 192 → 384 → 512 → 512   （越来越大）
```

含义：浅层看"细节"（需要高分辨率），深层看"概念"（需要多通道）。

**模式二：特征提取 + 分类头**

```
输入 → [ features：卷积/池化堆叠 ] → [ classifier：全连接 ] → 类别得分
         "看图"                         "下判断"
```

本文件 4 个模型全是这个结构，`self.features` 和 `self.classifier` 的命名也统一。

---

## 3. 文件整体结构

```
labkit/image_models.py
├── (1-5)   docstring：说明这些是"教学适配版"
├── (6-7)   import torch.nn / torchvision.models
├── (10-20)  class SmallCNN          ← 手写
├── (23-40)  class AlexNetCIFAR      ← 手写
├── (43-49)  def   vgg_block         ← 手写（造积木的函数）
├── (52-67)  class VGG11CIFAR        ← 手写
├── (70-109) def   build_image_model ← 工厂：按名字/模式造模型
└── (112-114) def  parameter_count   ← 统计参数
```

### docstring 里那句关键提醒

```python
"""CIFAR-10 教学适配版，以及原生 torchvision 模型调用入口。

适配版保留网络家族的关键结构，调整输入端与分类头以直接处理 32×32 图像。
它们不等同于 ImageNet 上的原始模型，不能直接加载原始预训练权重。
"""
```

这句话很重要，原因见 [8.6](#86-为什么-cifar-版加载不了预训练权重)：CIFAR 适配版改了首层卷积的形状，`state_dict` 的 key 形状对不上，加载 ImageNet 权重会直接报错。

---

## 4. `SmallCNN` — 最小可运行 CNN

📄 位置：[`image_models.py:10-20`](../labkit/image_models.py#L10-L20)

### 4.1 代码

```python
class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))
```

### 4.2 逐层输出形状（实测）

```
Conv2d(3→32, 3×3, p=1)     → (1, 32, 32, 32)
MaxPool2d(2)               → (1, 32, 16, 16)
Conv2d(32→64, 3×3, p=1)    → (1, 64, 16, 16)
MaxPool2d(2)               → (1, 64, 8, 8)
Conv2d(64→128, 3×3, p=1)   → (1, 128, 8, 8)
AdaptiveAvgPool2d(1)       → (1, 128, 1, 1)
flatten(1)                 → (1, 128)
Linear(128→10)             → (1, 10)
```

### 4.3 三个核心组件

| 组件 | 一句话作用 | 在 SmallCNN 中的体现 |
|---|---|---|
| **卷积 `Conv2d`** | 用可学习小窗口扫描图片，检测局部模式 | 3→32→64→128，通道递增表示模式越来越抽象 |
| **池化 `Pooling`** | 缩小特征图，降低计算量，增加平移不变性 | `MaxPool2d(2)` 减半；`AdaptiveAvgPool2d(1)` 全局汇总 |
| **全连接 `Linear`** | 把特征映射到类别得分 | 128→10 一个权重矩阵 |

### 4.4 `AdaptiveAvgPool2d(1)` 为什么好用

不管输入是 `(1,128,8,8)` 还是 `(1,128,16,16)`，它都输出 `(1,128,1,1)`——**整张特征图求平均，压成一个数**。

好处：**网络对输入尺寸不敏感**。想换成 64×64 的图，这个模型照样能跑（VGG11CIFAR 就不行，见 6.5）。

### 4.5 `flatten(1)` 里的 `1`

```python
self.features(x).flatten(1)
#                      ↑ start_dim=1，从第 1 维开始展平
```

```
(1, 128, 1, 1) --flatten(1)--> (1, 128)
 ↑  ↑
 │  └── 从这一维开始展平
 └── 第 0 维是 batch，保留不动
```

**如果不加 `start_dim=1`**（即 `flatten()`），batch 维也会被压平，变成 `(128,)`，全连接层就报错了。

### 4.6 参数量（实测 94,538）

| 层 | 计算 | 参数量 |
|---|---|---:|
| Conv1 (3→32) | (3×3×3+1)×32 | 896 |
| Conv2 (32→64) | (3×3×32+1)×64 | 18,496 |
| Conv3 (64→128) | (3×3×64+1)×128 | 73,856 |
| FC (128→10) | (128+1)×10 | 1,290 |
| **合计** | | **94,538** |

> 注意：**98.6% 的参数在卷积层，分类头只占 1.36%**。这和 AlexNet（FC 占比高）形成对比。

### 4.7 为什么叫 "Small"

| 对比 | SmallCNN | ResNet18(cifar) |
|---|---:|---:|
| 卷积层 | 3 | 17 |
| 参数量 | 94,538 | 11,173,962 |
| 准确率 | ~70% | ~93% |
| 训练速度 | 最快 | 中等 |

它是"最小可行模型"：概念完整、跑得快、结果能看，**适合第一次跑通流程**。

---

## 5. `AlexNetCIFAR` — 第一个"深"网络

📄 位置：[`image_models.py:23-40`](../labkit/image_models.py#L23-L40)

### 5.1 历史背景：为什么它重要

2012 年 ImageNet 竞赛，AlexNet 把 top-5 错误率从 26.2% 刷到 15.3%，碾压第二名。它确立了四件至今仍是标配的事：

| 贡献 | 现在还是标配吗 |
|---|---|
| 用 **ReLU** 替代 sigmoid/tanh | ✅ 是（解决了深层梯度消失） |
| 用 **Dropout** 抑制过拟合 | ✅ 是 |
| 用 **数据增强** 扩大训练集 | ✅ 是（对应 `image_transforms`） |
| 用 **GPU** 训练 | ✅ 是 |

### 5.2 代码

```python
class AlexNetCIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 五个卷积层，ReLU、池化和 Dropout；小图不采用 11×11 / stride=4。
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 192, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(192, 384, 3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(256, 512), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(512, num_classes))
```

### 5.3 逐层输出形状（实测）

```
Conv2d(3→64,   3×3, p=1)   → (1,  64, 32, 32)
MaxPool2d(2)               → (1,  64, 16, 16)   ↓1/2
Conv2d(64→192, 3×3, p=1)   → (1, 192, 16, 16)
MaxPool2d(2)               → (1, 192,  8,  8)   ↓1/2
Conv2d(192→384,3×3, p=1)   → (1, 384,  8,  8)   ← 尺寸不变
Conv2d(384→256,3×3, p=1)   → (1, 256,  8,  8)   ← 尺寸不变
Conv2d(256→256,3×3, p=1)   → (1, 256,  8,  8)
MaxPool2d(2)               → (1, 256,  4,  4)   ↓1/2
AdaptiveAvgPool2d(1)       → (1, 256,  1,  1)
Linear(256→512)            → (1, 512)
Linear(512→10)             → (1, 10)
```

**注意第 3、4 个卷积之间没有池化**：连续两个卷积在相同分辨率上做更细的特征加工。这种"同尺度堆叠"思路后来被 VGG/GoogLeNet 发扬光大。

### 5.4 `Dropout(0.5)` 的原理

**训练时**：每个神经元以 50% 概率被关闭，剩下的输出乘 `1/(1-p) = 2` 保持期望不变（inverted dropout）。

```
训练 (model.train()):
  输入  → [1.0, 2.0, 0.5, 3.0]
  掩码  → [ 1,   0,   1,   0 ]   ← 随机生成
  输出  → [2.0, 0.0, 1.0, 0.0]   ← 1.0×2 = 2.0,  0.5×2 = 1.0

评估 (model.eval()):
  输出  → [1.0, 2.0, 0.5, 3.0]   ← 什么都不做
```

**为什么能防过拟合？**

1. 每次训练都是"不同的子网络"在学，相当于训练了指数多个共享权重的子模型，最后做集成平均。
2. 阻止神经元之间形成"共谋"（某个神经元只在另一个特定神经元存在时才有用）。

> ⚠️ **这就是 `run_epoch` 里 `model.train(training)` 那行的意义。** SmallCNN 没有 Dropout，忘了切换没影响；AlexNet 忘了 `model.eval()` 会让验证准确率明显偏低。

### 5.5 与原始 AlexNet 的差异

| | 原始（ImageNet） | AlexNetCIFAR | 为什么改 |
|---|---|---|---|
| 输入 | 224×224 | 32×32 | CIFAR 图就这么小 |
| 首层卷积 | **11×11, stride=4** | **3×3, stride=1, p=1** | 大核大步长在 32×32 上一扫就没了大部分信息 |
| 参数 | 约 6,000 万 | **238 万** | 小 25 倍 |
| 分类头 | Flatten→4096→4096→1000 | AvgPool→256→512→10 | 省参数 |

#### 为什么首层不能照抄 11×11？

```
224×224 的图：11×11 窗口只覆盖 0.24% 面积 → 合理
 32×32  的图：11×11 窗口覆盖 11.8% 面积 → 一个窗口就盖住近 1/9 的图
             stride=4 时输出位置只有 (32-11)/4+1 = 6 个 → 信息损失严重
```

### 5.6 参数量分解（实测 2,388,298）

| 层 | 计算 | 参数量 | 占比 |
|---|---|---:|---:|
| Conv1 (3→64) | (3×3×3+1)×64 | 1,792 | 0.08% |
| Conv2 (64→192) | (3×3×64+1)×192 | 110,784 | 4.64% |
| Conv3 (192→384) | (3×3×192+1)×384 | 663,936 | 27.80% |
| Conv4 (384→256) | (3×3×384+1)×256 | **884,992** | **37.06%** |
| Conv5 (256→256) | (3×3×256+1)×256 | 590,080 | 24.71% |
| FC1 (256→512) | (256+1)×512 | 131,584 | 5.51% |
| FC2 (512→10) | (512+1)×10 | 5,130 | 0.21% |
| **合计** | | **2,388,298** | 100% |

> 📌 **规律**：越靠后的卷积层，因为 `C_in × C_out` 都在涨，参数量最大。这也是后来 VGG/ResNet 改用小卷积核、砍掉 FC 层的原因。

---

## 6. `vgg_block` + `VGG11CIFAR` — 模块化积木

📄 位置：[`image_models.py:43-49`](../labkit/image_models.py#L43-L49)（`vgg_block`）、[`52-67`](../labkit/image_models.py#L52-L67)（`VGG11CIFAR`）

### 6.1 `vgg_block` 代码

```python
def vgg_block(n_convs, in_channels, out_channels):
    layers = []
    for _ in range(n_convs):
        layers.extend([nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.ReLU()])
        in_channels = out_channels
    layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)
```

### 6.2 逐行拆解

```python
layers = []                       # 空列表，准备装层
```

```python
for _ in range(n_convs):          # 循环 n_convs 次（_ 表示"这个变量我不用"）
    layers.extend([Conv2d, ReLU])
    in_channels = out_channels    # ← 关键！
```

#### 🔑 `in_channels = out_channels` 为什么必须有？

因为一个 block 里有**多个连续卷积**，上一层的输出是下一层的输入：

```
block 有 2 个卷积，in=128, out=256:

  第 1 次循环: Conv2d(128 → 256)    ← 用最初的 128
               in_channels = 256    ← 更新！

  第 2 次循环: Conv2d(256 → 256)    ← 用更新后的 256 ✓
```

**没有这行**，第二次循环会写成 `Conv2d(128 → 256)`，通道数对不上，运行时直接报错。这是"循环建网络"最容易漏的一行。

#### 🔑 `extend` vs `append`

```python
layers.append([Conv, ReLU])   # → [[Conv, ReLU], ...]   嵌套列表，后面会出错
layers.extend([Conv, ReLU])   # → [Conv, ReLU, ...]     正确
```

`extend` 把列表里的元素**逐个**加进去。

#### 🔑 为什么 `MaxPool2d` 放在循环外

VGG 的定义就是"先堆卷积，最后一次性下采样"：

```
┌───────────────────────────┐
│ Conv3×3 → ReLU            │
│ Conv3×3 → ReLU   (n_convs 个) │
│ ...                       │
│ MaxPool2d(2)   ← 出口     │
└───────────────────────────┘
```

好处：同分辨率下能做更深的特征提取。坏处：池化前的大特征图要吃显存。

#### 🔑 `nn.Sequential(*layers)` 里的星号

```python
nn.Sequential(*layers)      # * 把列表"拆开"当位置参数传
```

等价于 `nn.Sequential(Conv, ReLU, Conv, ReLU, MaxPool)`。`Sequential` 不接受列表参数，所以必须解包。

### 6.3 `VGG11CIFAR` 代码与配置

```python
class VGG11CIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        blocks, in_channels = [], 3
        # VGG11 的卷积块配置：1、1、2、2、2 个卷积层。
        for n_convs, out_channels in [(1, 64), (1, 128), (2, 256), (2, 512), (2, 512)]:
            blocks.append(vgg_block(n_convs, in_channels, out_channels))
            in_channels = out_channels
        self.features = nn.Sequential(*blocks)
        # 32×32 经过五次池化后为 1×1，使用较小的分类头。
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, num_classes))
```

| Block | `n_convs` | 通道变化 | 该块卷积数 | 累计卷积数 |
|---|---:|---|---:|---:|
| 1 | 1 | 3 → 64 | 1 | 1 |
| 2 | 1 | 64 → 128 | 1 | 2 |
| 3 | 2 | 128 → 256 | 2 | 4 |
| 4 | 2 | 256 → 512 | 2 | 6 |
| 5 | 2 | 512 → 512 | 2 | 8 |

8 个卷积 + 3 个全连接 = **11 层** → 名字里的 "11"
（VGG16 = 13 卷积 + 3 FC；VGG19 = 16 卷积 + 3 FC）

> 这里的 `for` 循环外层也有一行 `in_channels = out_channels`，作用和 `vgg_block` 内部完全一样：把上一个 block 的输出通道，当作下一个 block 的输入通道。

### 6.4 逐层输出形状（实测）

```
Block1: Conv(3→64)      → (1,  64, 32, 32)  → Pool → (1,  64, 16, 16)
Block2: Conv(64→128)    → (1, 128, 16, 16)  → Pool → (1, 128,  8,  8)
Block3: Conv(128→256)   → (1, 256,  8,  8)
        Conv(256→256)   → (1, 256,  8,  8)  → Pool → (1, 256,  4,  4)
Block4: Conv(256→512)   → (1, 512,  4,  4)
        Conv(512→512)   → (1, 512,  4,  4)  → Pool → (1, 512,  2,  2)
Block5: Conv(512→512)   → (1, 512,  2,  2)
        Conv(512→512)   → (1, 512,  2,  2)  → Pool → (1, 512,  1,  1)
Flatten                 → (1, 512)
Linear(512→512)         → (1, 512)
Linear(512→10)          → (1, 10)
```

### 6.5 ⚠️ 五层池化 = 输入尺寸锁死 32×32

这就是代码里那行注释的含义：

```python
# 32×32 经过五次池化后为 1×1，使用较小的分类头。
```

```
32 → 16 → 8 → 4 → 2 → 1      五次池化刚好到 1×1
```

**如果输入是 224×224**：

```
224 → 112 → 56 → 28 → 14 → 7
Flatten 后是 512 × 7 × 7 = 25088 维
但 Linear 写的是 (512, 512) → 维度不匹配，直接报错！
```

三种模型的鲁棒性对比：

| 模型 | 收尾方式 | 对输入尺寸 |
|---|---|---|
| SmallCNN | `AdaptiveAvgPool2d(1)` + `flatten` | ✅ 任意尺寸都能跑 |
| AlexNetCIFAR | `AdaptiveAvgPool2d(1)` + `flatten` | ✅ 任意尺寸 |
| VGG11CIFAR | `nn.Flatten()` + 固定 `Linear(512,512)` | ❌ **必须 32×32** |

> 💡 **工程技巧**：想让网络支持任意输入尺寸，最后接 `AdaptiveAvgPool2d(1)`；想锁死尺寸追求极致速度，用 `Flatten` + 固定 FC。

### 6.6 VGG 的核心哲学：小卷积核堆叠

为什么用 3×3 而不用 5×5、7×7？

**① 感受野相同，参数更少**

两个 3×3 的感受野等于一个 5×5：

```
2 个 3×3:  参数 = 2 × (3×3×C×C) = 18C²
1 个 5×5:  参数 =     5×5×C×C   = 25C²      ← 多 39%
```

**② 非线性更多**

```
两个 3×3:  ReLU → Conv → ReLU → Conv    （2 次非线性）
一个 5×5:  ReLU → Conv                  （1 次非线性）
```

非线性越多，表达能力越强。

**③ 感受野递推公式（了解即可）**

```
RF_n = RF_{n-1} + (k_n - 1) × ∏_{i<n} s_i

两个 3×3、stride=1:  RF = 1 + 2 + 2 = 5 ✓
```

### 6.7 参数量分解（实测 9,488,266）

| 部分 | 参数量 | 占比 |
|---|---:|---:|
| `features`（全部卷积） | 9,220,480 | **97.18%** |
| `classifier`（两层 FC） | 267,786 | 2.82% |
| **合计** | **9,488,266** | 100% |

其中单层最大的：

```
Block4 第一个卷积:  3×3×256×512 + 512 = 1,180,160
Block4 第二个卷积:  3×3×512×512 + 512 = 2,359,808   ← 单层 236 万
Block5 两个卷积:    各 2,359,808
```

> 这就是 VGG **显存占用大、训练慢**的根源：参数几乎全在卷积层，而且越往后越大。

---

## 7. `ResNet18` — 残差网络

📄 位置：[`image_models.py:88-92`](../labkit/image_models.py#L88-L92)（在 `build_image_model` 里的 CIFAR 适配）

### 7.1 背景：深度网络的"退化问题"

直觉上"越深越好"，但实验发现：

```
   训练错误率
        │
  20%   │  ●  ── 56 层网络（更深）反而更差！
        │    ●
  10%   │ ●
        │   ● ── 20 层网络
        └──────────────────── 迭代次数
```

**注意：这不是过拟合**（训练误差也变大了），而是**优化困难**——深层网络的梯度传不回去。

### 7.2 核心思想：残差学习

```
普通网络:  H(x) = F(x)          直接学目标映射
ResNet:    H(x) = F(x) + x      学"增量"，再把输入加回去
```

#### 真实源码（从 `torchvision/models/resnet.py` 读取）

```python
def forward(self, x):
    identity = x                    # ① 保存输入

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
        identity = self.downsample(x)   # ② 形状变了要先调整

    out += identity                 # ③ 关键：把输入加回来
    out = self.relu(out)

    return out
```

那个 `out += identity` 就是 **shortcut connection / skip connection（跳连接）**。

### 7.3 为什么 `+ x` 能让梯度传回去？

对 `out = F(x) + x` 求导：

```
∂out/∂x = ∂F(x)/∂x + 1
                     ↑
              这个 +1 是关键
```

即使 `∂F/∂x` 很小（深层网络常见，接近 0），梯度仍然 ≈ `1`，**不会消失**。

图示：

```
没有残差:  x → [Conv][BN][ReLU][Conv][BN] → 输出
           梯度要穿过所有层才能回去，层数越多衰减越狠

有残差:    x ──────────────────────────────┐
            └→ [Conv][BN][ReLU][Conv][BN] ──┴→ (+) → 输出
                                                 ↑
            梯度可以从这条"旁路"直接流回
```

正因为有残差连接，ResNet 才能堆到 18/34/50/101/152 层而性能持续提升（VGG 堆到 19 层就难训了）。

### 7.4 CIFAR 适配的三处修改

```python
net = models.resnet18(weights=None)
net.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
net.maxpool = nn.Identity()
net.fc = nn.Linear(net.fc.in_features, 10)
```

#### 修改①：首层 7×7 stride 2 → 3×3 stride 1

```python
net.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
```

| | 原始 | 适配后 | 效果 |
|---|---|---|---|
| 卷积核 | 7×7 | 3×3 | 保留细节 |
| 步长 | 2 | 1 | **不下采样** |
| 输出尺寸 | 32/2 = 16 | 32 | 32×32 全保留 |
| 参数量 | 3×64×49 = 9,408 | 3×64×9 = 1,728 | 少 7,680 |

> 📌 **`bias=False` 为什么？** 因为紧接着就是 `BatchNorm2d`。BN 公式 `γ(x-μ)/σ + β` 里的 `β` 已经承担了偏置作用，卷积的 bias 会被完全抵消。所以"卷积 + BN"的标准做法是**不加 bias**。

#### 修改②：去掉首个最大池化

```python
net.maxpool = nn.Identity()
```

`nn.Identity()` 是"什么都不做"的层（输入原样输出）。

原始 ResNet 的 `maxpool` 是 `MaxPool2d(kernel_size=3, stride=2, padding=1)`，把 112×112 直接砍到 56×56。**但对 32×32 的小图，这一刀砍掉 3/4 信息**，所以跳过。

#### 修改③：替换分类头

```python
net.fc = nn.Linear(net.fc.in_features, 10)
```

`in_features` = 512（ResNet18 的固定设计）。**这里用 `net.fc.in_features` 而不是硬编码 512** 是个好习惯：换成 ResNet50（`in_features=2048`）这行不用改。

### 7.5 整体结构（实测）

```
 1. conv1    Conv2d(3, 64, 3, s=1, p=1, bias=False)
 2. bn1      BatchNorm2d(64)
 3. relu     ReLU
 4. maxpool  Identity                       ← 被替换掉
 5. layer1   Sequential   2 个 BasicBlock,  64 通道, 32×32
 6. layer2   Sequential   2 个 BasicBlock, 128 通道, 16×16
 7. layer3   Sequential   2 个 BasicBlock, 256 通道,  8×8
 8. layer4   Sequential   2 个 BasicBlock, 512 通道,  4×4
 9. avgpool  AdaptiveAvgPool2d(1)
10. fc       Linear(512, 10)
```

**"18" 怎么来的**：`1(conv1) + 2×8(8 个 block × 每个 2 卷积) + 1(fc) = 18`

#### 各 stage 分辨率变化（实测追踪）

```
conv1        → (1,  64, 32, 32)
layer1 (×2)  → (1,  64, 32, 32)      尺寸不变
layer2 (×2)  → (1, 128, 16, 16)      第 1 个 block stride=2 下采样
layer3 (×2)  → (1, 256,  8,  8)
layer4 (×2)  → (1, 512,  4,  4)
avgpool      → (1, 512,  1,  1)
fc           → (1, 10)
```

### 7.6 `downsample` 是什么

当 block 的输入输出**形状不一致**时（通道 64→128、尺寸 32→16），`out += identity` 没法直接相加。这时需要一个 **1×1 卷积 + BN** 把 `identity` 变换到同形状：

```
layer2 第 1 个 block:
    输入 (1, 64, 32, 32)

    主路径: Conv3×3(s=2) → (1,128,16,16) → Conv3×3 → (1,128,16,16)
    旁路:   1×1 Conv(s=2) + BN          → (1,128,16,16)   ← downsample
                                                     ↓
    相加:   (1,128,16,16) + (1,128,16,16) ✓
```

实测追踪里能看到这三个 1×1 卷积：

```
Conv2d in=64  out=128 k=1 s=2 p=0  → (1, 128, 16, 16)
Conv2d in=128 out=256 k=1 s=2 p=0  → (1, 256,  8,  8)
Conv2d in=256 out=512 k=1 s=2 p=0  → (1, 512,  4,  4)
```

### 7.7 参数量分解（实测 11,173,962）

| 模块 | 参数量 | 占比 |
|---|---:|---:|
| conv1 (3→64, 3×3, 无 bias) | 1,728 | 0.02% |
| bn1 | 128 | 0.00% |
| layer1 (2 个 block) | 147,968 | 1.32% |
| layer2 (2 个 block) | 525,568 | 4.70% |
| layer3 (2 个 block) | 2,099,712 | 18.79% |
| **layer4** (2 个 block) | **8,393,728** | **75.12%** |
| fc (512→10) | 5,130 | 0.05% |
| **合计** | **11,173,962** | 100% |

逐 block 明细（实测）：

```
layer1: block0 = 73,984  (无 downsample)
        block1 = 73,984  (无 downsample)
layer2: block0 = 230,144 (有 downsample)
        block1 = 295,424
layer3: block0 = 919,040 (有 downsample)
        block1 = 1,180,672
layer4: block0 = 3,673,088 (有 downsample)
        block1 = 4,720,640
```

> 📌 **75% 的参数集中在 layer4**（512×512 通道）。这也是为什么 ResNet 的深层部分最吃显存。

### 7.8 CIFAR 版 vs 官方版差多少

| | CIFAR 适配版 | 官方版（10 类） | 差值 |
|---|---:|---:|---:|
| 参数量 | 11,173,962 | 11,181,642 | **7,680** |

差值来源精确对应首层卷积：

```
官方:      3 × 64 × 7 × 7 = 9,408
CIFAR 版:  3 × 64 × 3 × 3 = 1,728
差值:                      7,680  ✓
```

**参数量几乎一样，但 CIFAR 版效果好得多**——因为它没有在小图上做两次粗暴下采样（stride=2 + maxpool），保住了空间分辨率。

---

## 8. `build_image_model` — 模型工厂

📄 位置：[`image_models.py:70-109`](../labkit/image_models.py#L70-L109)

### 8.1 为什么要工厂函数

调用方只写一句（[`train_image.py:38`](../train_image.py#L38)）：

```python
model = build_image_model(args.model, args.variant, args.pretrained, args.freeze_backbone)
```

好处：

| 好处 | 说明 |
|---|---|
| **单点入口** | 训练/预测/评估脚本都调它，不用各写一份 |
| **校验集中** | 所有非法组合在一个地方报错 |
| **易扩展** | 加新模型只需加一个分支 |

### 8.2 五道"守卫"（全部实测通过）

```python
if name not in {"smallcnn", "alexnet", "vgg11", "resnet18"}:
    raise ValueError("不支持的图像模型: " + name)
if variant not in {"cifar", "torchvision"}:
    raise ValueError("variant 必须是 cifar 或 torchvision")
if pretrained and variant != "torchvision":
    raise ValueError("预训练权重只适用于原生 torchvision 版本")
if freeze_backbone and not pretrained:
    raise ValueError("冻结骨干需要使用预训练权重")
if name == "smallcnn":
    if variant != "cifar" or pretrained or freeze_backbone:
        raise ValueError("SmallCNN 仅支持 cifar 模式")
```

| 非法组合 | 为什么非法 |
|---|---|
| `name="lenet"` | 不支持的模型名 |
| `variant="xx"` | 只有两种输入管线 |
| `pretrained=True, variant="cifar"` | CIFAR 适配版形状变了，加载不了 ImageNet 权重 |
| `freeze_backbone=True`（无预训练） | 冻结随机初始化的网络毫无意义 |
| `smallcnn + torchvision` | 手写模型没有官方 224 版本 |

#### 🔑 第 4 条特别值得体会

**冻结骨干 + 随机初始化 = 训练不动**。因为被冻结的参数永远是随机的，你只训练最后几层，等于"用一堆噪声提取特征"，几乎必然失败。这个检查拦下了一个会静默失败的实验。

### 8.3 `variant="cifar"` 分支

```python
if name == "smallcnn":
    ...
    return SmallCNN()
if variant == "cifar":
    if name == "alexnet":
        return AlexNetCIFAR()
    if name == "vgg11":
        return VGG11CIFAR()
    net = models.resnet18(weights=None)          # ← ResNet 改造官方实现
    net.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    net.fc = nn.Linear(net.fc.in_features, 10)
    return net
```

**为什么前三个自己写，ResNet 却改造官方实现？**

| 原因 | 说明 |
|---|---|
| 结构复杂 | 18 层 + 残差 + BN，手写容易出错 |
| 改造点少 | 只需改 3 行 |
| 保证正确性 | 复用官方测试过的实现 |

这是很好的**工程判断**：能复用就复用，只在必要处打补丁。

### 8.4 `variant="torchvision"` 分支

```python
builders = {
    "alexnet": (models.alexnet, models.AlexNet_Weights),
    "vgg11": (models.vgg11, models.VGG11_Weights),
    "resnet18": (models.resnet18, models.ResNet18_Weights),
}
builder, weights_enum = builders[name]
net = builder(weights=weights_enum.DEFAULT if pretrained else None)
```

#### 新版 API：用 `weights=` 而不是 `pretrained=`

```python
# ❌ 老写法（已废弃，会有 warning）
models.resnet18(pretrained=True)

# ✅ 新写法
models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
```

**为什么改？** 一个模型可能有**多套**权重（不同数据、不同版本、不同精度），用枚举更明确：

```python
models.ResNet18_Weights.DEFAULT          # 推荐的一套（IMAGENET1K_V1）
models.ResNet18_Weights.IMAGENET1K_V1
models.ResNet18_Weights.verify(...)      # 还带校验功能
```

#### `weights=None` = 随机初始化

```python
net = builder(weights=weights_enum.DEFAULT if pretrained else None)
```

**重要**：`--variant torchvision` 不带 `--pretrained` 时，模型是**随机初始化**的。它只是"结构一样、参数随机"的网络，和著名的预训练模型没有任何关系。

### 8.5 `freeze_backbone` — 顺序问题（代码特意注释了）

```python
if freeze_backbone:
    for param in net.parameters():
        param.requires_grad = False
# 先冻结旧参数，再替换最后一层，新分类器保持可训练。
if name == "resnet18":
    net.fc = nn.Linear(net.fc.in_features, 10)
else:
    net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, 10)
```

#### 🔑 为什么必须"先冻结、后替换"

```python
# ❌ 错误顺序：
net.classifier[-1] = nn.Linear(4096, 10)   # 先换新层
for p in net.parameters():
    p.requires_grad = False                 # ← 新层也被冻住，整个网络不训练

# ✅ 正确顺序（代码写法）：
for p in net.parameters():
    p.requires_grad = False                 # 先冻结全部
net.classifier[-1] = nn.Linear(4096, 10)   # 新层默认 requires_grad=True ✓
```

#### 实测效果（在 torchvision vgg11 上模拟）

```
before: total = 132,863,336                          ← 原始 1000 类
        head: Linear(in_features=4096, out_features=1000)

after : total = 128,807,306   trainable = 40,970     ← 换成 10 类
        head: Linear(in_features=4096, out_features=10)
        trainable ratio: 0.0318%                      ← 只有万分之三可训练！
```

**只剩 4 万个可训练参数（一个分类头），其余 1.28 亿全部冻结** —— 这就是迁移学习微调的典型形态：用预训练特征提取器，只重训最后的分类器。

#### 🔑 `net.classifier[-1]` 为什么可行

torchvision 的 `alexnet` / `vgg11` 的 `classifier` 是 `nn.Sequential`，最后一个是全连接层：

```python
# AlexNet 的 classifier（7 个元素）:
Sequential(
  (0): Dropout(0.5)
  (1): Linear(9216, 4096)
  (2): ReLU()
  (3): Dropout(0.5)
  (4): Linear(4096, 4096)
  (5): ReLU()
  (6): Linear(4096, 1000)    ← 替换这个
)
```

`Sequential` 支持整数下标赋值，所以 `net.classifier[-1] = ...` 就是"换掉最后一层"。

ResNet 不行，因为它的分类器是独立属性 `net.fc`（不是 Sequential）。

### 8.6 为什么 CIFAR 版加载不了预训练权重

```python
# 官方 ResNet18 的 state_dict:
'conv1.weight': torch.Size([64, 3, 7, 7])     # 7×7
# CIFAR 版的 conv1:
'conv1.weight': torch.Size([64, 3, 3, 3])     # 3×3
```

`load_state_dict` 遇到形状不匹配会直接报 `size mismatch`。

这就是文件开头 docstring 提醒的事：

> "它们不等同于 ImageNet 上的原始模型，不能直接加载原始预训练权重。"

**唯一理论上可行的特例**是 VGG 的特征层（全是 3×3，形状一致），但也没人这么用——因为输入分布不同，直接搬过来意义不大。

---

## 9. `parameter_count` — 参数统计

📄 位置：[`image_models.py:112-114`](../labkit/image_models.py#L112-L114)

```python
def parameter_count(model):
    return {"total": sum(p.numel() for p in model.parameters()),
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}
```

### `numel()` 是什么

**num**ber of **el**ements（元素个数）：

```python
x = torch.zeros(64, 3, 7, 7)
x.numel()      # 64×3×7×7 = 9,408
x.shape        # torch.Size([64, 3, 7, 7])
```

### 为什么区分 `total` 和 `trainable`

| 场景 | total | trainable |
|---|---|---|
| 从头训练 | 相等 | 相等 |
| 冻结骨干微调 | 128,807,306 | 40,970 |
| 部分冻结 | 相等 | 小于 total |

📄 [`train_image.py:47-48`](../train_image.py#L47-L48) 的优化器也做了同样的过滤：

```python
optimizer = torch.optim.SGD([x for x in model.parameters() if x.requires_grad],
                            lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
```

不过滤也能跑（PyTorch 会忽略冻结参数的更新），但会浪费优化器状态内存（SGD 的 momentum buffer）。**过滤掉是好习惯**。

### 输出在哪

📄 [`train_image.py:42-45`](../train_image.py#L42-L45) 会打印：

```json
"parameters": {
  "total": 11173962,
  "trainable": 11173962
}
```

这个数字用来**预估显存和训练时间**。

---

## 10. 实测数据汇总

### 10.1 四个 CIFAR 模型

| 模型 | total 参数 | 顶层模块分解 |
|---|---:|---|
| smallcnn | 94,538 | features 93,248 (98.64%) / classifier 1,290 (1.36%) |
| alexnet | 2,388,298 | features 2,251,584 (94.28%) / classifier 136,714 (5.72%) |
| vgg11 | 9,488,266 | features 9,220,480 (97.18%) / classifier 267,786 (2.82%) |
| resnet18 | 11,173,962 | conv1 1,728 / bn1 128 / layer1 147,968 / layer2 525,568 / layer3 2,099,712 / layer4 8,393,728 / fc 5,130 |

### 10.2 官方 torchvision 模型（10 类，224×224 输入）

| 模型 | 参数量 | 说明 |
|---|---:|---|
| alexnet | 57,044,810 | CIFAR 版的 **24 倍** |
| vgg11 | 128,807,306 | CIFAR 版的 **13.6 倍** |
| resnet18 | 11,181,642 | 和 CIFAR 版几乎相同 |

### 10.3 验证环境

| 项 | 值 |
|---|---|
| torch | 2.5.1+cu121 |
| torchvision | 0.20.1+cu121 |
| Python | (随 torch 环境) |

---

## 11. 四模型横向对比与选型

### 11.1 结构演进脉络

```
SmallCNN          AlexNetCIFAR        VGG11CIFAR          ResNet18
(3 卷积)     →    (5 卷积)       →    (8 卷积 + 积木) →   (17 卷积 + 残差)
94,538 参数       2,388,298 参数      9,488,266 参数      11,173,962 参数

思路： 能跑通   →   更深 + Dropout  →  模块化重复     →  跳连接解决退化
```

### 11.2 关键设计点对比

| 特性 | SmallCNN | AlexNetCIFAR | VGG11CIFAR | ResNet18 |
|---|---|---|---|---|
| 卷积层数 | 3 | 5 | 8 | 17 |
| Dropout | ❌ | ✅ 2×0.5 | ✅ 1×0.5 | ❌（用 BN） |
| BatchNorm | ❌ | ❌ | ❌ | ✅ 每个卷积后 |
| 残差连接 | ❌ | ❌ | ❌ | ✅ |
| 收尾方式 | MaxPool + AdaptiveAvg | MaxPool + AdaptiveAvg | **只能 MaxPool** | stride=2 下采样 |
| 输入尺寸依赖 | 任意 | 任意 | **必须 32×32** | 任意（32 最佳） |
| 分类头 | 1 层 FC | 2 层 FC | 2 层 FC | 1 层 FC |

### 11.3 怎么选

| 你的目的 | 推荐 |
|---|---|
| 第一次跑通流程 | `--model smallcnn --smoke`（最快看到结果） |
| 体会"加深网络"的效果 | smallcnn → alexnet → vgg11 依次跑，对比曲线 |
| 想要最好的 CIFAR 成绩 | `resnet18`（性价比最高） |
| 验证"残差连接有用" | 对比 vgg11 和 resnet18 的训练曲线 |
| 体验迁移学习 | `--model resnet18 --variant torchvision --pretrained --freeze-backbone`（需联网下载权重） |

### 11.4 运行命令示例

```bash
# 快速冒烟测试（128/64/64 样本，只验证流程能通）
python train_image.py --model smallcnn --smoke --run-dir runs/smoke

# 正式训练 SmallCNN
python train_image.py --model smallcnn --epochs 20 --run-dir runs/smallcnn

# 正式训练 ResNet18
python train_image.py --model resnet18 --epochs 20 --run-dir runs/resnet18

# 迁移学习微调（需联网）
python train_image.py --model resnet18 --variant torchvision \
    --pretrained --freeze-backbone --run-dir runs/resnet18_ft
```

---

## 12. 常见坑位清单

| # | 坑 | 现象 | 解法 |
|---|---|---|---|
| 1 | 给 `VGG11CIFAR` 喂非 32×32 输入 | `Flatten` 后维度不对，`Linear` 报错 | 固定 32×32，或改用 `AdaptiveAvgPool2d(1)` |
| 2 | 给 CIFAR 适配版加载 ImageNet 权重 | `size mismatch for conv1.weight` | 改用 `variant="torchvision"` |
| 3 | `freeze_backbone` 写在替换分类头之后 | 新分类头也被冻结，loss 不下降 | 先冻结再替换（代码已正确） |
| 4 | `freeze_backbone=True` 但没 `--pretrained` | 守卫拦下（好事） | 加上 `--pretrained` |
| 5 | 评估时忘了 `model.eval()` | Dropout 还在丢，验证准确率偏低且抖动 | 用 `run_epoch`，别手写循环 |
| 6 | 用极小 batch 训 ResNet（如 2） | BN 统计量不准，准确率明显变差 | batch ≥ 32，默认 64 就挺好 |
| 7 | 手写 ResNet `conv1` 忘了 `bias=False` | 参数略多，且与 BN 冗余 | 加 `bias=False` |
| 8 | 以为 `weights=None` 也是"预训练" | 以为有迁移效果，其实全随机 | 明确加 `--pretrained` |
| 9 | 替换 `classifier[-1]` 时硬编码 4096 | 换 VGG13/16 就报错 | 用 `net.classifier[-1].in_features` |
| 10 | `flatten()` 忘了 `start_dim=1` | batch 维被压平，形状错 | 写 `flatten(1)` |
| 11 | `vgg_block` 里漏了 `in_channels = out_channels` | 多卷积 block 通道数对不上 | 补上这行 |
| 12 | 循环建网络时用了 `append` 而非 `extend` | 出现嵌套列表，`Sequential` 报错 | 用 `extend` |

---

## 13. 自测题

<details>
<summary><b>Q1. SmallCNN 的参数量是多少？哪一层最多？</b></summary>

总共 **94,538**。最多的是 Conv3（64→128），`(3×3×64+1)×128 = 73,856`，占 78%。

</details>

<details>
<summary><b>Q2. 为什么 SmallCNN 和 AlexNetCIFAR 能用任意尺寸输入，而 VGG11CIFAR 不行？</b></summary>

前者用 `AdaptiveAvgPool2d(1)` 把任意尺寸的特征图压成 1×1，所以后续 `Linear(128→10)` 的输入维数固定。

VGG11CIFAR 用 `nn.Flatten()` 直接把 `(N, 512, H, W)` 拉平，`H×W` 必须等于 1，才能和固定的 `Linear(512, 512)` 匹配。而只有 32×32 输入经过 5 次池化才正好是 1×1。

</details>

<details>
<summary><b>Q3. 残差连接为什么能缓解梯度消失？用一句话和一行公式说明。</b></summary>

因为 `out = F(x) + x` 的导数是 `∂F/∂x + 1`，那个 `+1` 给梯度留了一条不衰减的通路。

</details>

<details>
<summary><b>Q4. `net.maxpool = nn.Identity()` 这行在干什么？为什么要这么做？</b></summary>

把 ResNet 原本的首个最大池化（3×3, stride=2）替换成"什么都不做"。

因为 32×32 的小图经不起 stride=2 下采样（会砍掉 3/4 信息），去掉它能让网络在 32×32 分辨率上做更多加工。

</details>

<details>
<summary><b>Q5. 为什么必须先冻结参数、再替换分类头？反了会怎样？</b></summary>

替换后的新分类头 `requires_grad` 默认是 `True`。如果先替换再统一冻结，新层也被冻住，整个网络没有任何可训练参数，loss 永远不变。

</details>

<details>
<summary><b>Q6. `vgg_block` 里 `in_channels = out_channels` 这行删掉会怎样？</b></summary>

只有当 `n_convs >= 2` 时才暴露问题：第二个卷积会写成 `Conv2d(原in_channels → out_channels)`，而实际输入通道已经是 `out_channels`，运行时抛出通道不匹配的错误。

</details>

<details>
<summary><b>Q7. ResNet18 的参数量中，哪个部分占比最大？大概多少？</b></summary>

`layer4`（512 通道的那两个 block），约 **8,393,728**，占 **75.12%**。

</details>

<details>
<summary><b>Q8. `weights=None` 和 `weights=...DEFAULT` 有什么区别？</b></summary>

`None` = 随机初始化，参数是随机的，没有任何预训练知识。
`DEFAULT` = 下载在 ImageNet 上训练好的权重，具备迁移学习能力。

</details>

<details>
<summary><b>Q9. 两个 3×3 卷积和一个 5×5 卷积相比，参数量差多少？</b></summary>

```
2 个 3×3:  18C²
1 个 5×5:  25C²
```

两个 3×3 **少 28%**（`18/25 = 0.72`），感受野相同，但非线性更多。

</details>

<details>
<summary><b>Q10. 如果要把 VGG11CIFAR 改成能吃 64×64 输入，最少改哪里？</b></summary>

64×64 经 5 次池化变成 2×2，`Flatten` 后是 `512×2×2 = 2048` 维。所以把 `Linear(512, 512)` 改成 `Linear(2048, 512)` 即可。

更稳妥的做法是把 `nn.Flatten()` 换成 `nn.AdaptiveAvgPool2d(1)` + `nn.Flatten()`，这样任意尺寸都能吃。

</details>

---

## 附录 A：一键验证脚本

把下面内容存成 `check_image_models.py`（放在 `d2l_pytorch_labs/` 下），运行 `python check_image_models.py`，可以复现本笔记里所有数字：

```python
"""打印每个图像模型的参数量分解和逐层输出形状。"""
import torch
import torch.nn as nn

from labkit.image_models import build_image_model, parameter_count


def trace(model, shape=(1, 3, 32, 32)):
    """用 forward hook 记录每个关键层的输出形状。"""
    lines, hooks = [], []

    def make_hook():
        def fn(module, inputs, output):
            cfg = ""
            if isinstance(module, nn.Conv2d):
                cfg = (f"in={module.in_channels} out={module.out_channels} "
                       f"k={module.kernel_size[0]} s={module.stride[0]} p={module.padding[0]}")
            lines.append(f"  {type(module).__name__:<20} {cfg:<36} -> {tuple(output.shape)}")
        return fn

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.MaxPool2d, nn.AdaptiveAvgPool2d, nn.Linear)):
            hooks.append(m.register_forward_hook(make_hook()))

    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(*shape))
    for h in hooks:
        h.remove()
    return out, lines


def main():
    for name in ["smallcnn", "alexnet", "vgg11", "resnet18"]:
        model = build_image_model(name, "cifar")
        stats = parameter_count(model)
        out, lines = trace(model)
        print("=" * 76)
        print(f"[{name}] total={stats['total']:,} trainable={stats['trainable']:,} out={tuple(out.shape)}")
        print("  --- 顶层模块参数 ---")
        for mod_name, child in model.named_children():
            cnt = sum(p.numel() for p in child.parameters())
            print(f"  {mod_name:<10} {type(child).__name__:<18} {cnt:>12,}")
        print("  --- 逐层输出形状 ---")
        for line in lines:
            print(line)


if __name__ == "__main__":
    main()
```

---

## 附录 B：它在训练流程中的位置

```
train_image.py
    │
    ├─ seed_everything(args.seed)                  固定随机种子
    ├─ device = choose_device(args.device)         选 GPU / CPU
    │
    ├─ model = build_image_model(...)      ←────── 本笔记的文件
    ├─ bundle = load_image_data(...)               数据 + 增强
    ├─ config = dict(..., parameters=parameter_count(model))   ←── 本笔记的统计
    │
    ├─ model.to(device)
    ├─ optimizer = SGD(可训练参数, lr, momentum=0.9, weight_decay)
    ├─ scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    │
    ├─ history = fit(...)                        训练 + 验证 + 保存 best.pt
    ├─ metrics = final_test(...)                 用最佳模型评测试集
    └─ save_run_plots(...)                       画曲线
```

关键联动：

| 位置 | 关联 |
|---|---|
| [`train_image.py:38`](../train_image.py#L38) | 调用工厂造模型 |
| [`train_image.py:42`](../train_image.py#L42) | 把参数量写进 `config.json` |
| [`train_image.py:47`](../train_image.py#L47) | 只把 `requires_grad=True` 的参数交给优化器 |
| [`engine.py:33`](../labkit/engine.py#L33) | `model.train(training)` 切换 Dropout/BN —— **对 AlexNetCIFAR 和 ResNet18 是必需的** |

---

## 附录 C：术语表

| 术语 | 全称 / 中文 | 一句话解释 |
|---|---|---|
| CNN | Convolutional Neural Network / 卷积神经网络 | 用卷积核提取图像特征的网络 |
| Conv2d | 2D Convolution | 二维卷积层，可学习的滑动窗口 |
| kernel / 卷积核 | — | 卷积层里的那个小窗口（本文件都是 3×3） |
| padding | 填充 | 在图像边缘补 0，`padding=1` 让尺寸不变 |
| stride | 步长 | 卷积核每次滑动几格 |
| channel | 通道 | 特征图的"厚度"；RGB 是 3，深层网络可以到 512 |
| pooling | 池化 | 缩小特征图；`MaxPool2d` 取最大，`AvgPool` 取平均 |
| AdaptiveAvgPool | 自适应平均池化 | 不管输入多大都压成指定尺寸（这里是 1×1） |
| ReLU | Rectified Linear Unit | `max(0, x)`，把负数变成 0 |
| Dropout | 随机失活 | 训练时随机关掉一部分神经元，防过拟合 |
| BatchNorm | 批归一化 | 按 batch 统计量归一化，让训练更稳更快 |
| Residual / skip connection | 残差 / 跳连接 | `out += identity`，给梯度留通路 |
| logits | 原始得分 | 模型最后输出的、未归一化的类别得分 |
| Softmax | 归一化指数 | 把 logits 变成概率（`CrossEntropyLoss` 内部自动做） |
| FLOPs | 浮点运算次数 | 衡量计算量的指标（参数量 ≠ 计算量） |
| epoch | 轮次 | 把整个训练集完整过一遍 |
| batch | 批 | 一次前向传播处理的样本数 |
| overfitting | 过拟合 | 训练集表现好、验证集表现差 |

---

## 学习路线建议

```
① 先看 SmallCNN，彻底搞懂"卷积 → ReLU → 池化 → 全连接"这条链路
        ↓
② 看 AlexNetCIFAR，理解"加深 + Dropout"带来的变化
        ↓
③ 看 vgg_block，理解"模块化/复用"的工程思想
        ↓
④ 看 ResNet18，理解"残差连接"为什么是革命性的
        ↓
⑤ 回到 build_image_model，理解工程上的封装与校验
        ↓
⑥ 动手：跑 train_image.py --smoke，再看 history.csv 的曲线
```

---

*笔记生成时间：2026-10-28 ｜ 基于实测数据，所有数字可复现（见附录 A）*
