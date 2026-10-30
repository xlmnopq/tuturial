"""CIFAR-10 教学适配版，以及原生 torchvision 模型调用入口。

适配版保留网络家族的关键结构，调整输入端与分类头以直接处理 32×32 图像。
它们不等同于 ImageNet 上的原始模型，不能直接加载原始预训练权重。
"""
import torch.nn as nn #PyTorch 的神经网络模块，nn.Module、nn.Conv2d、nn.ReLU
from torchvision import models #PyTorch 官方提供的经典模型（AlexNet、VGG、ResNet 等）


class SmallCNN(nn.Module): #小卷积网络（SmallCNN），用于 CIFAR-10 分类,继承自 nn.Module
    def __init__(self, num_classes=10): # 构造函数 nn.Sequential简化网络结构的定义和前向传播过程
        super().__init__() #调用父类 nn.Module 的初始化
        self.features = nn.Sequential(  #特征提取层，包含 3 个卷积层和 2 个最大池化层
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), #第一层输出：(1, 32, 16, 16) — 32 张 16×16 的特征图
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),#输入 (1, 32, 16, 16) — 32 张 16×16 图
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)) #nn.AdaptiveAvgPool2d(1) — 自适应平均池化，不管输入多大，都压成 1×1，从(1, 128, 8, 8) → (1, 128, 1, 1)
        self.classifier = nn.Linear(128, num_classes)# # 128 → 10

    def forward(self, x): #前向传播：特征提取 → flatten(1) 展平 → 全连接分类
        return self.classifier(self.features(x).flatten(1))


class AlexNetCIFAR(nn.Module): #AlexNet 模型，用于 CIFAR-10 分类,继承自 nn.Module
    def __init__(self, num_classes=10):
        super().__init__()
        #输入：32×32 图像，通道数：3
        # 五个卷积层，ReLU、池化和 Dropout；小图不采用 11×11 / stride=4。
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), #通道数：3 → 64，卷积核：3×3，填充：1，池化：2×2，-输出：(1, 64, 16, 16)
            nn.Conv2d(64, 192, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),#输出：(1, 192, 8, 8)
            nn.Conv2d(192, 384, 3, padding=1), nn.ReLU(),#输出：(1, 384, 8, 8)
            nn.Conv2d(384, 256, 3, padding=1), nn.ReLU(),#输出：(1, 256, 8, 8)
            #第 3、4 个卷积之间没有池化——连续两个卷积层在同样分辨率上做更细的特征加工（同尺度堆叠）
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), #输出：(1, 256, 4, 4)
            nn.AdaptiveAvgPool2d(1)) #输出：(1, 256, 1, 1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(256, 512), nn.ReLU(), 
            nn.Dropout(0.5), nn.Linear(512, num_classes))
            #每个神经元以 50% 概率被"关闭"（输出置 0），且剩下的神经元输出要乘 1/(1-p) = 2 保持期望不变
            #防止过拟合，提高模型泛化能力
    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


def vgg_block(n_convs, in_channels, out_channels): # VGG 卷积块，包含 n_convs 个卷积层和一个最大池化层
    layers = [] # 卷积层列表
    for _ in range(n_convs):
        layers.extend([nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.ReLU()])
        in_channels = out_channels
    layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers) # 卷积层 + 最大池化层  ，打包装箱（*解包）
    #① 空列表 → ② 循环加卷积 → ③ 更新通道状态 → ④ 循环外加池化 → ⑤ 装箱返回。
    #返回类似：Sequential(Conv2d, ReLU, Conv2d, ReLU, MaxPool2d)

class VGG11CIFAR(nn.Module):
    def __init__(self, num_classes=10): 
        super().__init__()
        blocks, in_channels = [], 3 # 卷积块列表，输入通道数为 3（RGB 图像）
        # VGG11 的卷积块配置：1、1、2、2、2 个卷积层。
        for n_convs, out_channels in [(1, 64), (1, 128), (2, 256), (2, 512), (2, 512)]:
            blocks.append(vgg_block(n_convs, in_channels, out_channels))
            in_channels = out_channels  #vgg_block是局部变量，要在这里再写一次
        self.features = nn.Sequential(*blocks)
        # 32×32 经过五次池化后为 1×1，使用较小的分类头。
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


def build_image_model(name, variant="cifar", pretrained=False, freeze_backbone=False):
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
        return SmallCNN()
    if variant == "cifar":
        if name == "alexnet":
            return AlexNetCIFAR()
        if name == "vgg11":
            return VGG11CIFAR()
        net = models.resnet18(weights=None)
        net.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        net.maxpool = nn.Identity()
        net.fc = nn.Linear(net.fc.in_features, 10)
        return net
    # 官方实现的实际调用方式，weights=None 表示随机初始化。
    builders = {
        "alexnet": (models.alexnet, models.AlexNet_Weights),
        "vgg11": (models.vgg11, models.VGG11_Weights),
        "resnet18": (models.resnet18, models.ResNet18_Weights),
    }
    builder, weights_enum = builders[name]
    net = builder(weights=weights_enum.DEFAULT if pretrained else None)
    if freeze_backbone:
        for param in net.parameters():
            param.requires_grad = False
    # 先冻结旧参数，再替换最后一层，新分类器保持可训练。
    if name == "resnet18":
        net.fc = nn.Linear(net.fc.in_features, 10)
    else:
        net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, 10)
    return net


def parameter_count(model):
    return {"total": sum(p.numel() for p in model.parameters()),
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}
