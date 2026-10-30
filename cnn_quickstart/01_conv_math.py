"""
01_conv_math.py —— 不依赖 PyTorch，手写卷积 / 池化 / 展平 / 全连接，看清 CNN 每一步的张量变化。

只用到 Python 标准库，任何 Python 3.8+ 都能直接运行：
    python 01_conv_math.py

学习目标
1. 卷积核到底在算什么：一个 3×3 的"检测器"在图上滑动，逐位置做加权求和。
2. 输出尺寸公式：out = floor((in + 2*padding - kernel) / stride) + 1
3. padding 的作用：让边缘像素也能被计算，并且保持尺寸不变。
4. 通道机制：输入 C_in 个通道 → 每个卷积核有 C_in 个"层"，所有通道加权求和后输出一个通道；C_out 个卷积核 → C_out 个通道。
5. 池化：只做局部统计（取最大/求平均），不学参数，主要用来降尺寸。
"""
import math


# --------------------------------------------------------------------------- #
# 1. 单个二维卷积（单通道）
# --------------------------------------------------------------------------- #
def conv2d_single(matrix, kernel, padding=0, stride=1): # 
    """单通道二维卷积。matrix: H×W 的嵌套列表；kernel: k×k 的嵌套列表。"""
    H, W = len(matrix), len(matrix[0])
    k = len(kernel)
    if padding > 0:
        # 四周补 0：在原图上下左右各加 padding 行/列
        padded = [[0.0] * (W + 2 * padding) for _ in range(padding)]
        for row in matrix:
            padded.append([0.0] * padding + list(row) + [0.0] * padding)
        padded += [[0.0] * (W + 2 * padding) for _ in range(padding)]
        matrix, H, W = padded, H + 2 * padding, W + 2 * padding

    OH = (H - k) // stride + 1  # 输出边长公式（向下取整）
    OW = (W - k) // stride + 1
    out = [[0.0] * OW for _ in range(OH)]
    for i in range(OH):
        for j in range(OW):
            # 卷积核覆盖的输入区域：行 i*stride ... i*stride+k-1，列同理
            r0, c0 = i * stride, j * stride
            s = 0.0
            for u in range(k):
                for v in range(k):
                    s += matrix[r0 + u][c0 + v] * kernel[u][v]  # 对应位置相乘（element-wise）
            out[i][j] = s  # 再求和 → 这就是"加权求和"
    return out


def conv2d_multi(matrix, kernels, padding=0, stride=1):
    """多输出通道卷积：kernels 是 [kernel_1, kernel_2, ...]，每个卷积核产生一个输出通道。"""
    return [conv2d_single(matrix, k, padding, stride) for k in kernels]


def conv2d_channels(channels, kernels, padding=0, stride=1):
    """多输入 + 多输出通道卷积（PyTorch 的 Conv2d 就是干这个的）。

    channels: [C_in 张 H×W 特征图]
    kernels : [C_out 个卷积核]，每个卷积核是 [C_in 张 k×k 权重]
    规则：第 o 个输出通道 = sum_{c} conv(channels[c], kernels[o][c])
    所以一个 Conv2d(3, 64, 3) 的权重形状是 (64, 3, 3, 3)。
    """
    outputs = []
    for kernel_set in kernels:
        assert len(kernel_set) == len(channels), "每个卷积核的层数必须等于输入通道数"
        per_input = [conv2d_single(ch, w, padding, stride) for ch, w in zip(channels, kernel_set)]
        H, W = len(per_input[0]), len(per_input[0][0])
        acc = [[sum(per_input[c][i][j] for c in range(len(per_input)))
                for j in range(W)] for i in range(H)]
        outputs.append(acc)
    return outputs


# --------------------------------------------------------------------------- #
# 2. 池化
# --------------------------------------------------------------------------- #
def max_pool2d(matrix, k=2, stride=None):
    stride = stride or k
    H, W = len(matrix), len(matrix[0])
    OH, OW = (H - k) // stride + 1, (W - k) // stride + 1
    return [[max(matrix[i * stride + u][j * stride + v]
                 for u in range(k) for v in range(k))
             for j in range(OW)] for i in range(OH)]


def avg_pool2d(matrix, k=2, stride=None):
    stride = stride or k
    H, W = len(matrix), len(matrix[0])
    OH, OW = (H - k) // stride + 1, (W - k) // stride + 1
    return [[sum(matrix[i * stride + u][j * stride + v]
                 for u in range(k) for v in range(k)) / (k * k)
             for j in range(OW)] for i in range(OH)]


def relu(matrix):
    """逐元素：负数变 0。引入非线性，否则多层卷积等价于一层线性变换。"""
    return [[max(0.0, value) for value in row] for row in matrix]


# --------------------------------------------------------------------------- #
# 3. 用"亲手构造"的数据演示：边缘检测器
# --------------------------------------------------------------------------- #
def demo_edge_detector():
    print("=" * 70)
    print("演示 1：3×3 卷积核如何检测竖直边缘")
    print("=" * 70)

    # 6×6 的图：左三列 0，右三列 1 → 中间有一条竖直边缘
    image = [[0, 0, 0, 1, 1, 1]] * 6

    # Sobel 竖直边缘检测核：左右两侧权重相反
    sobel_x = [[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]]

    no_pad = conv2d_single(image, sobel_x, padding=0)
    same_pad = conv2d_single(image, sobel_x, padding=1)
    print("输入 6×6：")
    show(image)
    print("输出（padding=0，尺寸 4×4）：")
    show(no_pad, fmt="{:5.0f}")
    print("输出（padding=1，尺寸仍为 6×6，边界用 0 补齐）：")
    show(same_pad, fmt="{:5.0f}")
    print("→ 观察：只有中间一列出现了很大的正值，说明这个卷积核专门对'竖直边缘'响应。\n")


def demo_three_kernels():
    print("=" * 70)
    print("演示 2：三个不同的卷积核 = 三个特征图（C_out 的含义）")
    print("=" * 70)
    image = [[0, 0, 0, 1, 1, 1]] * 3 + [[1, 1, 1, 1, 1, 1]] * 3  # 右上角有竖直边缘，下方有水平边缘
    kernels = {
        "竖直边缘": [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        "水平边缘": [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        "平均模糊": [[1 / 9] * 3] * 3,
    }
    show(image, fmt="{:3.0f}", title="输入 6×6")
    for name, kernel in kernels.items():
        show(conv2d_single(image, kernel, padding=1), fmt="{:5.1f}", title=f"卷积核[{name}] 的输出")
    print("→ 每个卷积核学到/被设计成一种'模式检测器'；真实网络里这些权重是反向传播学出来的。\n")


def demo_channels():
    print("=" * 70)
    print("演示 3：多输入通道 → 多输出通道（对应 nn.Conv2d(3, 2, 3)）")
    print("=" * 70)
    # 3 个输入通道，各 4×4
    r = [[1] * 4 for _ in range(4)]
    g = [[0] * 4 for _ in range(4)]
    b = [[2] * 4 for _ in range(4)]
    # 2 个卷积核，每个卷积核有 3 层（对应 R/G/B）
    kernel_red_lover = [[[1, 0, 0], [0, 0, 0], [0, 0, 0]],   # R 层
                        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],   # G 层
                        [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]   # B 层
    kernel_blue_lover = [[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                         [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                         [[0, 0, 0], [0, 0, 1], [0, 0, 0]]]
    out = conv2d_channels([r, g, b], [kernel_red_lover, kernel_blue_lover], padding=1)
    print("输出通道数 =", len(out), "（= 卷积核个数）；每个输出尺寸 =", len(out[0]), "×", len(out[0][0]))
    show(out[0], fmt="{:3.0f}", title="输出通道 0（只对 R 通道的左上角加权）")
    show(out[1], fmt="{:3.0f}", title="输出通道 1（只对 B 通道的右下角加权）")
    print("→ 一个输出通道 = 它自己的卷积核在'所有输入通道'上的加权求和，所以 3 通道输入也能得到 1 通道输出。\n")


def demo_shapes():
    print("=" * 70)
    print("演示 4：尺寸公式速查 —— out = floor((in + 2p - k) / s) + 1")
    print("=" * 70)
    for name, inp, k, p, s in [
        ("Conv2d(3,32,3,padding=1)", 32, 3, 1, 1),   # 保持尺寸
        ("Conv2d(3,32,5,padding=2)", 32, 5, 2, 1),   # 保持尺寸
        ("Conv2d(3,32,3,stride=2,padding=1)", 32, 3, 1, 2),  # 下采样为一半
        ("Conv2d(3,32,3)", 32, 3, 0, 1),             # 每层少 2
        ("MaxPool2d(2)", 32, 2, 0, 2),               # 池化减半（教材写法：pool = 2）
        ("MaxPool2d(2)", 16, 2, 0, 2),
    ]:
        out = math.floor((inp + 2 * p - k) / s) + 1
        print(f"  输入 {inp:>3}  {name:<34} → 输出 {out:>3}")
    print("\n→ 教材里 \"pool=2\" 对应 PyTorch 的 nn.MaxPool2d(2)，即 k=2, stride=2。\n")


def demo_full_cnn_shapes():
    print("=" * 70)
    print("演示 5：跟着 SmallCNN 走一遍，看每一步张量形状（对应 labkit/image_models.py）")
    print("=" * 70)
    steps = [
        ("输入图像 (1, 3, 32, 32)", 3, 32),
        ("Conv2d(3,32,3,padding=1) → (1, 32, 32, 32)", 32, 32),
        ("MaxPool2d(2) → (1, 32, 16, 16)", 32, 16),
        ("Conv2d(32,64,3,padding=1) → (1, 64, 16, 16)", 64, 16),
        ("MaxPool2d(2) → (1, 64, 8, 8)", 64, 8),
        ("Conv2d(64,128,3,padding=1) → (1, 128, 8, 8)", 128, 8),
        ("AdaptiveAvgPool2d(1) → (1, 128, 1, 1)", 128, 1),
        ("flatten(1) → (1, 128)", 128, 1),
        ("Linear(128, 10) → (1, 10)  # 10 个类别的原始分数 logits", 10, 1),
    ]
    for text, _, _ in steps:
        print("   " + text)
    print("\n→ 规律：空间尺寸（长宽）不断变小，通道数不断变大，最后把 (C, H, W) 压成向量交给全连接层。\n")


def _demo_padding_effect():
    """补充：padding=0 时输出比输入小多少。"""
    image = [[1.0] * 5 for _ in range(5)]
    for padding in (0, 1, 2):
        print(f"   5×5 输入, 3×3 卷积核, padding={padding} → 输出 {len(conv2d_single(image, [[1] * 3] * 3, padding))}×{len(conv2d_single(image, [[1] * 3] * 3, padding))}")


def demo_padding():
    print("=" * 70)
    print("演示 4.5：padding 与输出尺寸的关系")
    print("=" * 70)
    _demo_padding_effect()
    print("→ 3×3 卷积核配 padding=1 就能'尺寸不变'，这是现代 CNN 的默认写法。\n")


def show(matrix, fmt="{:3.0f}", title=None):
    if title:
        print(f"  [{title}]")
    for row in matrix:
        print("    " + " ".join(fmt.format(v) for v in row))
    print()


if __name__ == "__main__":
    demo_edge_detector()
    demo_three_kernels()
    demo_channels()
    demo_shapes()
    demo_padding()
    demo_full_cnn_shapes()
    print("下一步：打开 02_pytorch_cnn.py，用 PyTorch 把同样的操作写成 5 行代码训练一个网络。")