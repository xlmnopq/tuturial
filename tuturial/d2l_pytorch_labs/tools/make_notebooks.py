"""从本项目的课程内容生成可编辑 Notebook，不需要额外 notebook 库。"""
import json
from pathlib import Path
from textwrap import dedent
from uuid import uuid4

ROOT = Path(__file__).absolute().parents[1]


def md(text):
    return {"cell_type": "markdown", "id": uuid4().hex[:8], "metadata": {},
            "source": dedent(text).strip() + "\n"}


def code(text):
    return {"cell_type": "code", "id": uuid4().hex[:8], "metadata": {},
            "execution_count": None, "outputs": [], "source": dedent(text).strip() + "\n"}


def write(name, cells):
    notebook = {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "PyTorch Labs", "language": "python", "name": "d2l-labs"},
        "language_info": {"name": "python", "version": "3.10"}}, "nbformat": 4, "nbformat_minor": 5}
    (ROOT / "notebooks" / name).write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")


BOOTSTRAP = '''
from pathlib import Path
import os, sys

# 从项目根目录、notebooks/ 或它们的子目录启动时都能找到 labkit。
candidates = [Path.cwd(), *Path.cwd().parents]
ROOT = next((p for p in candidates if (p / "labkit").is_dir()), None)
if ROOT is None:
    raise RuntimeError("请将整个课程目录解压后，从项目根目录启动 JupyterLab")
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
print("项目根目录:", ROOT)
'''

write("00_environment.ipynb", [
    md("""
    # 实验 00：运行环境与 GPU
    目标：确认正在使用的 Python 内核，检查 CUDA，完成一次张量反向传播。
    阅读：[D2L 5.6 GPU](https://zh.d2l.ai/chapter_deep-learning-computation/use-gpu.html)。
    安装步骤见项目 README，安装完成后应重启并选择正确的内核。
    """), code(BOOTSTRAP),
    code("from check_environment import main\nmain()"),
    md("""
    ## 张量、损失与梯度
    参数、输入必须处于相同设备。`backward()` 计算梯度，优化器的 `step()` 更新参数。
    """),
    code('''
    import torch
    from labkit.common import choose_device, seed_everything
    seed_everything(42)
    device = choose_device("auto")
    model = torch.nn.Linear(4, 2).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    x = torch.randn(8, 4, device=device)
    y = torch.tensor([0, 1] * 4, device=device)
    before = model.weight.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    loss = torch.nn.CrossEntropyLoss()(logits, y)
    loss.backward()
    optimizer.step()
    print("device =", device, "loss =", loss.item())
    print("参数确实更新:", not torch.equal(before, model.weight))
    '''),
    md("""
    ## 数据准备
    在项目终端执行 `python prepare_data.py --dataset all`。全套实验只使用 CIFAR-10 和 IMDB。
    教师也可以复制已经下载好的 `data/` 目录供学生离线使用。

    ## 检查题
    1. `nvidia-smi` 能显示 GPU，但 `torch.cuda.is_available()` 为 False 时，你会检查什么？
    2. `model.eval()` 和 `torch.no_grad()` 各自解决什么问题？
    3. 修改随机种子，观察初始损失是否改变。
    """)])


def image_notebook(filename, title, model_name, link, concept, exercises):
    cells = [md(f"""
        # {title}
        {concept}

        阅读：[对应教材章节]({link})。图像数据统一为 CIFAR-10。
        本实验默认是 32×32 小图教学适配版。原生 torchvision 版本的切换方法见后文。
        """), code(BOOTSTRAP),
        md("## 1. 实验配置\n先用 `SMOKE=True` 跑通，再设为 False 正式训练。小规模检查不能代表模型最终性能。"),
        code(f'''
        import inspect
        from datetime import datetime
        import torch
        from labkit.common import choose_device, seed_everything
        from labkit.data import load_image_data
        from labkit.image_models import build_image_model, parameter_count
        from labkit.engine import run_epoch, fit, final_test
        from labkit.plots import show_images, plot_history, plot_confusion, save_run_plots

        MODEL = "{model_name}"
        VARIANT = "cifar"
        PRETRAINED = False
        FREEZE_BACKBONE = False
        SMOKE = os.environ.get("LAB_SMOKE", "0") == "1"
        SEED = 42
        BATCH_SIZE = 64
        EPOCHS = 1 if SMOKE else 20
        LR = 0.01
        USE_AMP = False  # 可在完成基础实验后改为 True，比较显存与速度。
        DATA_DIR = ROOT / "data"
        RUN_DIR = ROOT / "runs" / (MODEL + "_" + VARIANT + datetime.now().strftime("_%Y%m%d_%H%M%S_%f"))
        seed_everything(SEED)
        device = choose_device("auto")
        print("device:", device, "SMOKE:", SMOKE, "结果目录:", RUN_DIR)
        '''),
        md("""
        ## 2. 数据加载、划分与预处理
        官方训练集按类别分成 45,000 训练 / 5,000 验证，官方测试集保持独立。
        训练图像使用随机裁剪与翻转，验证/测试只做固定预处理。
        数据加载代码在 `labkit/data.py`，运行下面一格查看实际规模与张量形状。
        """),
        code('''
        limits = [128, 64, 64] if SMOKE else [0, 0, 0]
        bundle = load_image_data(DATA_DIR, batch_size=BATCH_SIZE, seed=SEED,
                                 workers=0, variant=VARIANT, augment=True,
                                 max_train=limits[0], max_val=limits[1], max_test=limits[2])
        print(bundle.meta["sizes"])
        images, labels = next(iter(bundle.val))
        print("images:", images.shape, "labels:", labels.shape)
        show_images(images, labels, bundle.meta["classes"], bundle.meta["mean"], bundle.meta["std"])
        '''),
        md("## 3. 创建模型\n输出是 `[batch, 10]` 的 logits。`CrossEntropyLoss` 内部处理 log-softmax，训练前不手动 softmax。"),
        code('''
        model = build_image_model(MODEL, VARIANT, PRETRAINED, FREEZE_BACKBONE).to(device)
        print(model)
        print("参数量:", parameter_count(model))
        model.eval()
        with torch.no_grad():
            print("输出形状:", model(images[:2].to(device)).shape)
        '''),
        md("""
        ## 4. 阅读完整训练循环
        `run_epoch` 包含训练/评估模式切换、前向传播、损失、梯度、裁剪、参数更新与按样本统计指标。
        `fit` 每个 epoch 只在验证集上选模型，最后恢复最佳权重。以下代码直接显示当前实现，可去源文件修改。
        """), code("print(inspect.getsource(run_epoch))\nprint(inspect.getsource(fit))"),
        md("## 5. 训练与验证\n先把模型移动到 GPU，再创建优化器。第一次正式实验建议保留相同划分和超参数。"),
        code('''
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                                    lr=LR, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        config = {"task": "image", "model": MODEL, "variant": VARIANT,
                  "pretrained": PRETRAINED, "freeze_backbone": FREEZE_BACKBONE,
                  "seed": SEED, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
                  "lr": LR, "amp": USE_AMP, "smoke": SMOKE,
                  "parameters": parameter_count(model), "torch_version": str(torch.__version__)}
        history = fit(model, bundle, optimizer, device, RUN_DIR, config,
                       epochs=EPOCHS, amp=USE_AMP, grad_clip=5.0,
                       freeze_backbone=FREEZE_BACKBONE, scheduler=scheduler)
        plot_history(history)
        '''),
        md("## 6. 最终测试\n此时模型已经恢复到最佳验证 epoch。测试结果用于最终报告，不能据此反复调参。混淆矩阵的行是真实类别，列是预测类别。"),
        code('''
        metrics = final_test(model, bundle, device, RUN_DIR)
        save_run_plots(history, metrics, RUN_DIR)
        plot_confusion(metrics["confusion_matrix"], bundle.meta["classes"])
        print("accuracy:", metrics["accuracy"], "macro F1:", metrics["macro_f1"])
        '''),
        md("## 7. 可视化预测\n标题格式为“真实类别 / 预测类别”。对于自选图片，可使用项目中的 `predict.py`。"),
        code('''
        images, labels = next(iter(bundle.test))
        model.eval()
        with torch.no_grad():
            predictions = model(images.to(device)).argmax(1).cpu()
        show_images(images, labels, bundle.meta["classes"], bundle.meta["mean"],
                     bundle.meta["std"], predictions=predictions)
        '''),
        md("""
        ## 8. 扩展：原生 torchvision 模型
        重新运行实验前将 `VARIANT="torchvision"`、`PRETRAINED=True`，输入自动变为 224×224。
        `FREEZE_BACKBONE=True` 只训练新分类层。AlexNet / ResNet18 建议 batch size 16，VGG11 建议先设为 8。
        SmallCNN 不提供此模式。迁移学习实验单独比较，并报告额外使用了 ImageNet 预训练权重。

        命令行入口示例：
        ```bash
        python train_image.py --model resnet18 --variant torchvision --pretrained --freeze-backbone --batch-size 16 --epochs 5 --run-dir runs/resnet_transfer
        ```
        """), md("## 9. 实操练习\n" + exercises + "\n\n实验报告按 `docs/REPORT_TEMPLATE.md` 填写。所有数值来自自己的实际运行。")]
    write(filename, cells)


image_notebook("01_cifar10_pipeline.ipynb", "实验 01：CIFAR-10 与完整分类流程", "smallcnn",
    "https://zh.d2l.ai/chapter_computer-vision/image-augmentation.html",
    "用一个较小的 CNN 理解完整流程，然后在后续实验替换网络结构。",
    "1. 设为不做增广，与原配置对比验证曲线。\n2. 把学习率改为 0.001 和 0.1，解释 loss 的变化。\n3. 只观察训练准确率会遗漏什么问题？")
image_notebook("02_alexnet.ipynb", "实验 02：AlexNet 图像分类", "alexnet",
    "https://zh.d2l.ai/chapter_convolutional-modern/alexnet.html",
    "观察五个卷积层以及分类头中的 Dropout，理解小图版本对输入端与全连接层的调整。",
    "1. 在 `AlexNetCIFAR` 中比较 Dropout=0 和 0.5。\n2. 记录输入经过每个卷积/池化层后的形状。\n3. 对比默认版本与原生 torchvision 版本的参数量，说明差异。")
image_notebook("03_vgg11.ipynb", "实验 03：VGG11 与卷积块", "vgg11",
    "https://zh.d2l.ai/chapter_convolutional-modern/vgg.html",
    "通过重复的 3×3 卷积块搭建网络，使用 VGG11 的 1/1/2/2/2 卷积块配置。",
    "1. 修改一个 block 的卷积层数，比较参数量与训练耗时。\n2. 说明 32×32 图像经过五次 2×2 池化后为什么变为 1×1。\n3. 将 batch size 从 64 改为 16，记录吞吐率与显存。")
image_notebook("04_resnet18.ipynb", "实验 04：ResNet18 与残差连接", "resnet18",
    "https://zh.d2l.ai/chapter_convolutional-modern/resnet.html",
    "调用 torchvision 的 ResNet18，并为 CIFAR-10 调整输入卷积、初始池化和输出层。",
    "1. 查看 `model.layer2[0]`，解释下采样支路为什么需要 1×1 卷积。\n2. 比较 3×3 小图输入端与原生 7×7 输入端。\n3. 用同一划分比较 AlexNet、VGG11、ResNet18，至少报告准确率、macro F1 和参数量。")


def text_notebook(filename, title, model_name, chapter, concept, exercises):
    write(filename, [
        md(f"""
        # {title}
        {concept}
        阅读：[对应教材章节](https://zh.d2l.ai/chapter_natural-language-processing-applications/{chapter}.html)。
        全部文本实验使用 IMDB 英文影评，不更换数据集，不额外下载预训练词向量。
        """), code(BOOTSTRAP), md("## 1. 配置\n正式实验使用完整数据，`SMOKE=True` 仅检查流程。"),
        code(f'''
        from datetime import datetime
        import inspect
        import torch
        from labkit.common import choose_device, seed_everything
        from labkit.data import load_text_data, tokenize
        from labkit.text_models import build_text_model
        from labkit.image_models import parameter_count
        from labkit.engine import run_epoch, fit, final_test
        from labkit.plots import plot_history, plot_confusion, save_run_plots
        from labkit.inference import predict_text

        MODEL = "{model_name}"
        SMOKE = os.environ.get("LAB_SMOKE", "0") == "1"
        SEED, BATCH_SIZE = 42, 64
        MAX_LENGTH, MAX_VOCAB, MIN_FREQ = 256, 20000, 2
        EPOCHS, LR = (1 if SMOKE else 8), 0.001
        DATA_DIR = ROOT / "data"
        RUN_DIR = ROOT / "runs" / (MODEL + datetime.now().strftime("_%Y%m%d_%H%M%S_%f"))
        seed_everything(SEED)
        device = choose_device("auto")
        print("device:", device, "SMOKE:", SMOKE, "结果目录:", RUN_DIR)
        '''),
        md("""
        ## 2. 原始影评、分词、词表与 DataLoader
        标签 0=negative、1=positive。词表只统计训练集，`<pad>=0`、`<unk>=1`。
        序列先截断至最大长度，再按当前批次动态填充。文本 CNN 至少需要 5 个位置。
        """),
        code('''
        limits = [128, 64, 64] if SMOKE else [0, 0, 0]
        bundle = load_text_data(DATA_DIR, BATCH_SIZE, SEED, workers=0,
                                max_length=MAX_LENGTH, max_vocab=MAX_VOCAB, min_freq=MIN_FREQ,
                                max_train=limits[0], max_val=limits[1], max_test=limits[2])
        print("规模:", bundle.meta["sizes"], "词表大小:", len(bundle.vocab))
        sample = bundle.train.dataset.records[0]
        print("影评片段:", sample["text"][:350])
        print("标签:", sample["label"])
        print("分词:", tokenize(sample["text"])[:25])
        tokens, lengths, labels = next(iter(bundle.val))
        print("tokens:", tokens.shape, "真实长度:", lengths[:8], "labels:", labels[:8])
        print("特殊 token:", bundle.vocab.tokens[:2])
        '''),
        md("## 3. 模型结构\n词向量通过当前训练集学习。输出为两个类别的 logits，交叉熵之前不做 softmax。"),
        code('''
        model = build_text_model(MODEL, len(bundle.vocab)).to(device)
        print(model)
        print("参数量:", parameter_count(model))
        print(inspect.getsource(type(model)))
        model.eval()
        with torch.no_grad():
            logits = model(tokens[:2].to(device), lengths[:2].to(device))
        print("输出形状:", logits.shape)
        '''),
        md("## 4. 训练与验证\n阅读完整循环，注意验证阶段没有反向传播，梯度裁剪可以限制过大的梯度。"),
        code("print(inspect.getsource(run_epoch))\nprint(inspect.getsource(fit))"),
        code('''
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        config = {"task": "text", "model": MODEL, "seed": SEED,
                  "batch_size": BATCH_SIZE, "max_length": MAX_LENGTH,
                  "max_vocab": MAX_VOCAB, "min_freq": MIN_FREQ,
                  "epochs": EPOCHS, "lr": LR, "smoke": SMOKE,
                  "parameters": parameter_count(model), "torch_version": str(torch.__version__)}
        history = fit(model, bundle, optimizer, device, RUN_DIR, config,
                       epochs=EPOCHS, amp=False, grad_clip=1.0)
        plot_history(history)
        '''),
        md("## 5. 最终测试\n恢复最佳验证模型后，报告 accuracy、macro F1 和混淆矩阵。"),
        code('''
        metrics = final_test(model, bundle, device, RUN_DIR)
        save_run_plots(history, metrics, RUN_DIR)
        plot_confusion(metrics["confusion_matrix"], bundle.meta["classes"])
        print("accuracy:", metrics["accuracy"], "macro F1:", metrics["macro_f1"])
        '''),
        md("## 6. 从磁盘恢复模型并预测\n预测必须复用保存的词表和分词规则，不能临时创建新词表。低置信度或错误预测也是需要分析的结果。"),
        code('''
        examples = ["The story was moving and the acting was excellent.",
                    "The movie was boring and I would not recommend it.",
                    "The acting was great, but the plot was disappointing."]
        for text in examples:
            print(text)
            print(predict_text(RUN_DIR, text, str(device)))
        '''),
        md("## 7. 实操练习\n" + exercises + "\n\n与另一种文本模型比较时，保持种子、词表容量、最大长度、训练轮数一致。结果使用 `compare_runs.py` 汇总。")])


text_notebook("05_imdb_bilstm.ipynb", "实验 05：IMDB 与 BiLSTM 情感分类", "bilstm",
    "sentiment-analysis-rnn", "理解双向 LSTM 如何将变长影评编码为固定维度特征。",
    "1. 分别设置最大长度 128、256、512，观察验证效果与训练耗时。\n2. 解释为什么不能简单使用填充后最后一个位置的输出。\n3. 找出至少 5 条错分影评，分析否定、转折或长距离依赖。")
text_notebook("06_imdb_textcnn.ipynb", "实验 06：TextCNN 情感分类", "textcnn",
    "sentiment-analysis-cnn", "使用不同宽度的卷积核提取局部词组特征，再进行最大池化。",
    "1. 将卷积核改为只使用 (3,)，比较参数量与验证效果。\n2. 解释 TextCNN 为什么需要处理 PAD 对池化的影响。\n3. 与 BiLSTM 比较 macro F1、训练耗时和错误样本，避免只按单一指标判断。")
GEN_SETUP = '''
from datetime import datetime
import inspect
import torch
from IPython.display import display, Image as DisplayImage
from labkit.common import choose_device, seed_everything, save_json
from labkit.data import load_image_data
from labkit.generative_models import ConvVAE, DCGenerator, DCDiscriminator, dcgan_init, vae_loss
from labkit.generative import (fit_vae, fit_gan, vae_epoch, gan_train_step, gan_diagnostics,
                               save_vae_examples, save_grid, noise, plot_generative_history)
SMOKE = os.environ.get("LAB_SMOKE", "0") == "1"
SEED, LATENT_DIM, CHANNELS, BATCH_SIZE = 42, 64, 32, 128
device = choose_device("auto")
seed_everything(SEED)
limits = [128, 64, 64] if SMOKE else [0, 0, 0]
# 与分类实验共享相同的 CIFAR-10 划分，但生成任务不做随机增广。
bundle = load_image_data(ROOT / "data", BATCH_SIZE, SEED, augment=False,
                         max_train=limits[0], max_val=limits[1], max_test=limits[2])
print("device:", device, "数据规模:", bundle.meta["sizes"], "SMOKE:", SMOKE)
'''

write("07_cifar10_vae.ipynb", [
    md("""
    # 实验 07：CIFAR-10 卷积 VAE
    目标：重参数化、重构项与 KL 项、验证模型选择、测试和先验采样。
    参考：[VAE 原始论文](https://arxiv.org/abs/1312.6114)、[PyTorch VAE 示例](https://github.com/pytorch/examples/tree/main/vae)。
    本实现将 VAE 改为处理 CIFAR-10 的 32×32 RGB 卷积结构，不切换第三个数据集。
    """), code(BOOTSTRAP),
    md("## 1. 数据加载\n训练/验证/测试划分与其他 CIFAR-10 实验相同，标签在生成模型训练中不使用。输入统一归一化至 [-1,1]。"),
    code(GEN_SETUP),
    md("""
    ## 2. 模型与损失
    编码器输出 `mu`、`logvar`。重参数化使用 `z = mu + exp(0.5*logvar)*eps`。
    解码器以 Tanh 输出条件均值，使用固定标准差 0.1 的高斯像素似然。
    重构负对数似然在每张图的全部像素上求和，KL 在潜变量维度求和，然后都对 batch 求均值。
    `beta=1` 时为负 ELBO 的一次 Monte Carlo 估计。`beta!=1` 应称为 beta-VAE 目标。
    """),
    code('''
    EPOCHS = 1 if SMOKE else 30
    BETA, OBSERVATION_STD, LR = 1.0, 0.1, 0.001
    RUN_DIR = ROOT / "runs" / datetime.now().strftime("vae_%Y%m%d_%H%M%S_%f")
    config = {"task": "vae", "model": "conv_vae", "epochs": EPOCHS, "lr": LR,
              "latent_dim": LATENT_DIM, "channels": CHANNELS, "seed": SEED,
              "beta": BETA, "observation_std": OBSERVATION_STD, "smoke": SMOKE,
              "batch_size": BATCH_SIZE, "torch_version": str(torch.__version__)}
    model = ConvVAE(LATENT_DIM, CHANNELS).to(device)
    print(model)
    print(inspect.getsource(vae_loss))
    print(inspect.getsource(ConvVAE.reparameterize))
    '''),
    md("## 3. 完整训练与验证\n按验证目标选择最佳 epoch，保存到 best.pt。验证只访问官方训练集划分出的 5,000 张图，不访问测试集。"),
    code("print(inspect.getsource(vae_epoch))\nprint(inspect.getsource(fit_vae))"),
    code('''
    history = fit_vae(model, bundle, device, RUN_DIR, config)
    plot_generative_history(history, "vae", RUN_DIR / "learning_curves.png")
    '''),
    md("""
    ## 4. 最终测试与重构
    测试负 ELBO 使用固定种子的一次后验采样估计。重构 MSE / PSNR 使用 `decode(mu)`，在 [0,1] 显示空间计算。
    PSNR 为 `-10*log10(全部像素平均 MSE)`，它不能衡量先验随机生成的质量。
    连续高斯密度可以大于 1，含常数项的 NLL 可能为负。
    """),
    code('''
    metrics = vae_epoch(model, bundle.test, device, beta=BETA, observation_std=OBSERVATION_STD)
    save_json(RUN_DIR / "test_metrics.json", metrics)
    print(metrics)
    save_vae_examples(model, bundle.test, device, RUN_DIR, "test", SEED + 2000)
    # 上排原图，下排使用潜变量均值解码的重构图。
    display(DisplayImage(filename=str(RUN_DIR / "test_reconstruction.png")))
    '''),
    md("## 5. 先验生成\n从 N(0,I) 采样潜变量，显示解码器条件均值。生成阶段不需要编码器或原始输入图。"),
    code('''
    display(DisplayImage(filename=str(RUN_DIR / "test_samples.png")))
    print("独立进程生成命令:")
    print(f'python generate.py --run-dir "{RUN_DIR}" --seed 2026')
    '''),
    md("""
    ## 6. 实操练习
    1. 潜维度使用 16、64、128，比较重构和先验生成。
    2. 改变 beta，分别画出 KL 和重构项，记录潜变量是否遭到忽略。
    3. 使用两个训练图像的 mu 做线性插值并解码，观察中间变化。
    4. 解释“重构好”为什么不保证“先验采样好”。每次变更使用新的运行目录。
    """)])

write("08_cifar10_gan.ipynb", [
    md("""
    # 实验 08：CIFAR-10 DCGAN
    目标：理解 G / D 的交替更新、detach、固定噪声与训练不稳定。
    参考：[PyTorch DCGAN 教程](https://docs.pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)。
    本实验把卷积生成对抗网络改为 32×32 RGB 的 CIFAR-10，使用全部类别的混合图像，不输入类别标签。
    """), code(BOOTSTRAP), md("## 1. 数据加载\n沿用相同的 CIFAR-10 划分，图像范围 [-1,1] 与生成器 Tanh 输出一致。"),
    code(GEN_SETUP),
    md("""
    ## 2. 生成器与判别器
    G 用转置卷积把噪声变成图像。D 输出单个 logit，使用 `BCEWithLogitsLoss`，无需手动 sigmoid。
    D 的损失为真假两项的平均，G 的非饱和损失把生成图的目标设为 1。
    """),
    code('''
    EPOCHS = 1 if SMOKE else 50
    RUN_DIR = ROOT / "runs" / datetime.now().strftime("gan_%Y%m%d_%H%M%S_%f")
    config = {"task": "gan", "model": "dcgan32", "epochs": EPOCHS, "lr": 0.0002,
              "latent_dim": LATENT_DIM, "channels": CHANNELS, "seed": SEED,
              "batch_size": BATCH_SIZE, "smoke": SMOKE,
              "torch_version": str(torch.__version__)}
    generator = DCGenerator(LATENT_DIM, CHANNELS).to(device)
    discriminator = DCDiscriminator(CHANNELS).to(device)
    generator.apply(dcgan_init)
    discriminator.apply(dcgan_init)
    print(generator)
    print(discriminator)
    '''),
    md("""
    ## 3. 一次完整的对抗更新
    更新 D 时使用 `fake.detach()`，避免更新 G。更新 G 时冻结 D 的参数和 BN 状态，但保留 D 对输入的梯度。
    G 与 D 使用各自独立的优化器。
    """), code("print(inspect.getsource(gan_train_step))\nprint(inspect.getsource(fit_gan))"),
    md("## 4. 训练、验证诊断与固定噪声\n每轮保存同一组噪声的生成图。使用预先指定的训练轮数，保存 last.pt，不按 D 的 loss 宣称某轮的 G 最好。"),
    code('''
    history = fit_gan(generator, discriminator, bundle, device, RUN_DIR, config)
    plot_generative_history(history, "gan", RUN_DIR / "learning_curves.png")
    display(DisplayImage(filename=str(RUN_DIR / "previews" / f"epoch_{EPOCHS:03d}.png")))
    '''),
    md("""
    ## 5. 留出测试集诊断
    这里记录 D 对未参与训练的真实图像和独立噪声生成图的分数，以及两者的通道均值/标准差。
    **这些不是分类准确率或生成质量分数，不能替代 FID/KID 或覆盖率分析。**
    本入门包不额外下载特征网络，也不把像素统计冒充 FID。比较图像质量需进一步规范评估协议。
    """),
    code('''
    metrics = gan_diagnostics(generator, discriminator, bundle.test, device, LATENT_DIM, SEED + 2000)
    save_json(RUN_DIR / "test_metrics.json", metrics)
    print(metrics)
    generator.eval()
    with torch.no_grad():
        save_grid(generator(noise(64, LATENT_DIM, device, SEED + 3000)), RUN_DIR / "test_samples.png")
    display(DisplayImage(filename=str(RUN_DIR / "test_samples.png")))
    print(f'python generate.py --run-dir "{RUN_DIR}" --seed 2026')
    '''),
    md("""
    ## 6. 实操练习
    1. 比较不同 epoch 的固定噪声图，记录清晰度、重复样本和类型覆盖。
    2. 将 G 或 D 的学习率单独扩大，观察训练失衡。
    3. 解释 `detach()` 和 `requires_grad_(False)` 的不同作用。
    4. G loss 下降、D 输出接近 0.5，是否足以证明训练成功？给出原因。
    5. 用相同噪声种子生成两次，确认恢复的模型给出相同结果。
    """)])
print("Generated 9 notebooks")
