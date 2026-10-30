# 参考与实现说明

查阅日期：2026-10-21。代码围绕教材主题独立组织，未把原网站的全部实验复制到本项目。

## 教材

- [《动手学深度学习》中文第二版](https://zh.d2l.ai/index.html)
- [7.1 AlexNet](https://zh.d2l.ai/chapter_convolutional-modern/alexnet.html)
- [7.2 VGG](https://zh.d2l.ai/chapter_convolutional-modern/vgg.html)
- [7.6 ResNet](https://zh.d2l.ai/chapter_convolutional-modern/resnet.html)
- [13.1 图像增广](https://zh.d2l.ai/chapter_computer-vision/image-augmentation.html)
- [15.1 情感分析及数据集](https://zh.d2l.ai/chapter_natural-language-processing-applications/sentiment-analysis-and-dataset.html)
- [15.2 使用循环神经网络做情感分析](https://zh.d2l.ai/chapter_natural-language-processing-applications/sentiment-analysis-rnn.html)
- [15.3 使用卷积神经网络做情感分析](https://zh.d2l.ai/chapter_natural-language-processing-applications/sentiment-analysis-cnn.html)

## 数据

- [CIFAR-10 官方页](https://www.cs.toronto.edu/~kriz/cifar.html)：60,000 张 32×32 RGB 图像，50,000 官方训练 / 10,000 官方测试，10 个类别。原始 Python 包 MD5 为 `c58f30108f718f92721af3b95e74349a`。
- [MIT Open Learning 可选原始压缩包镜像](https://huggingface.co/datasets/MIT-OL-AI-D/cifar-10-python)：下载固定提交 `a48007227f9e2cd0af96175f4afb5ac4965e261b`，仍检查上述官方 MD5。不会改变样本来源与划分。
- [IMDB / Large Movie Review Dataset 官方页](https://ai.stanford.edu/~amaas/data/sentiment/)：25,000 官方训练 / 25,000 官方测试。参考 Maas et al., *Learning Word Vectors for Sentiment Analysis*, ACL 2011。
- IMDB SHA-1 `01ada507287d82875905620988597833ad4e0903` 与教材的原始数据注册项一致。仅使用原始压缩包中的 train/test 的 pos/neg，忽略无标签数据。

## PyTorch 官方资料

- [CIFAR-10 分类教程](https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [torchvision 模型与预训练权重](https://docs.pytorch.org/vision/stable/models.html)
- [AlexNet](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.alexnet.html)
- [VGG11](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vgg11.html)
- [ResNet18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)
- [历史版本安装命令](https://pytorch.org/get-started/previous-versions/)
- [DCGAN 教程](https://docs.pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
- [DCGAN 教程源文件](https://github.com/pytorch/tutorials/blob/main/beginner_source/dcgan_faces_tutorial.py)
- [PyTorch VAE 示例](https://github.com/pytorch/examples/tree/main/vae)

## 生成模型论文

- Kingma & Welling, [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)，预印本 2013，ICLR 2014。用于变分下界、对角高斯 KL 与重参数化。
- Goodfellow et al., [Generative Adversarial Nets](https://arxiv.org/abs/1406.2661)，NeurIPS 2014。用于对抗目标和非饱和生成器训练。
- Radford et al., [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)，ICLR 2016。用于卷积 G/D 的结构思路。

## 本包的教学取舍

- 默认分类模型保留对应网络家族的核心结构，为 32×32 图像调整输入端/分类头。原生 torchvision 模型另有可选入口，二者不混称。
- IMDB 不引入预训练词向量，不依赖已停止维护的旧版 torchtext 数据 API。
- BiLSTM 用真实序列长度打包，TextCNN 屏蔽填充区窗口。
- VAE 使用固定方差高斯像素密度，保留 NLL 常数项，明确 beta 与缩放方式。
- GAN 使用 BCEWithLogitsLoss，D 的真假损失取平均，G 使用非饱和目标。32×32 网络与教程的 64×64 人脸版本不同。
- 指标与图像必须由实际运行生成；本包不预填教材在其他数据集上的准确率。
