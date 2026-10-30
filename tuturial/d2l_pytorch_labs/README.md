# PyTorch 图像分类与情感分类实操

这套材料参考《动手学深度学习》第二版的教学主题，选择 **CIFAR-10** 和 **IMDB** 两个数据集贯穿全部实验。代码为独立教学实现，中文注释，不依赖 `d2l` 或 `torchtext`。主要面向带 NVIDIA GPU 的电脑或服务器。

## 实验安排

| 顺序 | Notebook | 学生完成的任务 | 教材对应章节 |
|---|---|---|---|
| 00 | `notebooks/00_environment.ipynb` | 检查 Python、PyTorch、CUDA，完成一次 GPU 反向传播 | 5.6 |
| 01 | `notebooks/01_cifar10_pipeline.ipynb` | 加载数据、固定划分、图像增广，训练 SmallCNN 基线 | 3.5、6、13.1 |
| 02 | `notebooks/02_alexnet.ipynb` | AlexNet 分类与 Dropout 实验 | 7.1 |
| 03 | `notebooks/03_vgg11.ipynb` | VGG 卷积块与网络深度实验 | 7.2 |
| 04 | `notebooks/04_resnet18.ipynb` | 使用 ResNet18，观察残差块 | 7.6 |
| 05 | `notebooks/05_imdb_bilstm.ipynb` | 分词、词表、变长批次、BiLSTM 情感分类 | 15.1、15.2 |
| 06 | `notebooks/06_imdb_textcnn.ipynb` | TextCNN 情感分类，与 BiLSTM 比较 | 15.3 |
| 07 | `notebooks/07_cifar10_vae.ipynb` | 卷积 VAE、重参数化、重构与生成 | VAE 原始论文 / PyTorch 示例 |
| 08 | `notebooks/08_cifar10_gan.ipynb` | DCGAN 交替训练、固定噪声与测试诊断 | PyTorch DCGAN 教程 |

每个正式实验都包含加载数据、训练、验证、保存最佳模型、最终测试和预测。Notebook 按单元格组织，公共实现位于 `labkit/`，训练循环完整公开，方便学生阅读与修改。也提供独立命令行脚本，便于服务器运行。

## 数据集与划分

| 数据集 | 任务 | 官方训练集再划分 | 官方测试集 |
|---|---|---|---|
| CIFAR-10 | 32×32 RGB 图像，10 类 | 45,000 训练 + 5,000 验证 | 10,000 |
| IMDB | 英文影评，负面/正面二分类 | 22,500 训练 + 2,500 验证 | 25,000 |

划分按类别分层、固定种子，保存在 `data/splits/`。所有图像模型使用同一份划分，两个文本模型也使用同一份划分。官方测试集不参与调参或选择最佳 epoch。IMDB 的额外无标签数据不使用，词表仅从实际训练子集建立。验证/测试图像不做随机增广。默认 `seed=42`。

数据来源：[CIFAR-10 官方](https://www.cs.toronto.edu/~kriz/cifar.html)、[IMDB 官方](https://ai.stanford.edu/~amaas/data/sentiment/)。IMDB 是英文任务，输入预测文本也应使用英文。数据集不会塞进代码压缩包，教师可课前下载一次并分发缓存目录。

## 安装与启动

建议使用独立的 Python 3.10 / 3.11 环境。先进入本文件所在目录。

```bash
python -m venv .venv
```

Windows PowerShell 激活：`.venv\Scripts\Activate.ps1`；Linux 激活：`source .venv/bin/activate`。

以下是一组便于统一课堂环境的固定版本，**并非最新版本声明**。CUDA 12.1 wheel 的安装方式来自 [PyTorch 官方历史版本页](https://pytorch.org/get-started/previous-versions/)。较新的 GPU 架构或学校现有环境请使用 [官方安装选择器](https://pytorch.org/get-started/locally/) 选择匹配组合。

```bash
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m ipykernel install --user --name d2l-labs --display-name "PyTorch Labs"
python check_environment.py
```

不需要额外安装 `torchaudio`。已有可用 GPU 环境可保留，只安装缺少的课程依赖。安装后的实际版本与本次验证范围见 `docs/VALIDATION.md`。

```bash
python prepare_data.py --dataset all
python -m jupyterlab
```

CIFAR-10 官方源较慢时可改用 `python prepare_data.py --dataset all --cifar-mirror`。该选项使用固定版本的高校镜像，并核验与官方完全相同的 MD5。

选择 `PyTorch Labs` 内核，先打开 `00_environment.ipynb`，再按顺序运行。Notebook 会定位项目根目录，并给每次运行建立独立结果目录。Windows / Jupyter 默认 `workers=0`，服务器脚本可设为 `--workers 4`。

课程默认将 PyTorch CPU 计算线程设为 4，避免小批量 GPU 实验的线程调度开销。需要时设置 `LAB_NUM_THREADS` 环境变量。固定种子有助于复现，但不保证跨设备和软件版本逐位相同。

## 先做一次小规模检查

`--smoke` 使用真实数据中的分层子集：128 训练、64 验证、64 测试，只运行 1 个 epoch。它用于检查数据管线、梯度更新、保存与加载，不提供有意义的模型性能结论。

```bash
python train_image.py --model resnet18 --smoke --run-dir runs/check_resnet
python train_text.py --model bilstm --smoke --run-dir runs/check_bilstm
```

Notebook 中将 `SMOKE=True` 可做相同检查，或启动前设置环境变量 `LAB_SMOKE=1`。正式实验应使用 `SMOKE=False`。

## 图像分类：正式实验

默认使用 **CIFAR 教学适配版**，直接处理 32×32 图像，从随机初始化开始训练。它们保留相应家族的关键结构，但并非原始 ImageNet 模型的完整复现：

| 模型 | 保留的结构 | 针对 CIFAR-10 的调整 |
|---|---|---|
| AlexNet | 5 层卷积、ReLU、池化、Dropout | 小卷积输入端，缩小全连接分类头 |
| VGG11 | 1/1/2/2/2 卷积块，5 次池化 | 512 维小分类头 |
| ResNet18 | torchvision 的 BasicBlock、[2,2,2,2] 残差阶段 | 3×3 stride=1 输入卷积，去掉初始池化，10 类输出 |

```bash
python train_image.py --model smallcnn --epochs 20 --run-dir runs/cnn
python train_image.py --model alexnet --epochs 20 --run-dir runs/alexnet
python train_image.py --model vgg11 --epochs 20 --run-dir runs/vgg11
python train_image.py --model resnet18 --epochs 20 --run-dir runs/resnet18
python compare_runs.py --runs runs/alexnet runs/vgg11 runs/resnet18 --output runs/image_comparison.csv
```

默认 batch size 64、SGD + momentum、余弦学习率调度。以上是可运行的统一起点，不承诺最优准确率。显存不足时改为 `--batch-size 16` 或 `32`。需要混合精度可加 `--amp`。训练时间取决于 GPU 和轮数，本材料不预填虚构的准确率。

### 扩展：调用 torchvision 原生模型与迁移学习

将输入改为 224×224，保留官方骨干网络并替换十分类输出。下面三个命令都会从 ImageNet 预训练权重开始，只训练新分类层。首次运行会下载权重，需要额外网络与磁盘空间。

```bash
python train_image.py --model alexnet --variant torchvision --pretrained --freeze-backbone --batch-size 16 --lr 0.01 --epochs 5 --run-dir runs/alexnet_transfer
python train_image.py --model vgg11 --variant torchvision --pretrained --freeze-backbone --batch-size 8 --lr 0.01 --epochs 5 --run-dir runs/vgg_transfer
python train_image.py --model resnet18 --variant torchvision --pretrained --freeze-backbone --batch-size 16 --lr 0.01 --epochs 5 --run-dir runs/resnet_transfer
```

移除 `--freeze-backbone` 可全模型微调，建议降低学习率，例如 `--lr 0.001`。移除 `--pretrained` 与 `--freeze-backbone` 可随机初始化原生模型。迁移学习利用了额外的 ImageNet 预训练数据，应与从零训练分表报告，不能直接把差异归因于网络结构。

原生用法示例：

```python
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 10)
```

AlexNet、VGG 对应 `model.classifier[-1]`。三种模型的完整处理见 `labkit/image_models.py`。

## 情感分类：正式实验

两个模型共用分词、词表容量、序列截断长度和 IMDB 数据划分。词向量从随机初始化训练，不额外引入第三个数据集或预训练词向量。`<pad>=0`、`<unk>=1`，最大长度默认 256，词表上限 20,000。

```bash
python train_text.py --model bilstm --epochs 8 --run-dir runs/bilstm
python train_text.py --model textcnn --epochs 8 --run-dir runs/textcnn
python compare_runs.py --runs runs/bilstm runs/textcnn --output runs/text_comparison.csv
```

BiLSTM 使用真实句长打包，并拼接两个方向的最终隐状态。TextCNN 使用 3、4、5 三种卷积核与最大池化，屏蔽完全处于填充区的卷积窗口。两种模型输出二分类 logits，损失使用 `CrossEntropyLoss`，训练前无需手动 softmax。

## 测试与预测

训练脚本会在训练结束后恢复验证集表现最好的权重，再执行最终测试。也可以在独立进程复测：

```bash
python evaluate.py --run-dir runs/resnet18
python evaluate.py --run-dir runs/bilstm
python predict.py --run-dir runs/resnet18 --image your_image.jpg
python predict.py --run-dir runs/bilstm --text "The story was moving and the acting was excellent."
```

`evaluate.py` 默认严格复用该运行保存的测试索引。仅在明确希望将 smoke 模型放到完整官方测试集上检查时使用 `--full-test`，这不会把 smoke 模型变成正式训练的模型。概率是模型输出分数，不代表已校准的可靠性。

## CIFAR-10 生成模型

VAE 与 GAN 继续使用同一份 CIFAR-10，完整说明见 [生成模型实验指南](docs/GENERATIVE_GUIDE.md)。

```bash
python train_vae.py --epochs 30 --run-dir runs/vae
python train_gan.py --epochs 50 --run-dir runs/gan
python evaluate_generative.py --run-dir runs/vae
python evaluate_generative.py --run-dir runs/gan
python generate.py --run-dir runs/vae --num-samples 64 --seed 2026
python generate.py --run-dir runs/gan --num-samples 64 --seed 2026
```

两者都支持 `--smoke`、`--offline`、`--device`。VAE 保存最佳验证模型、重构对照与先验生成图，GAN 保存最终 G/D、各轮固定噪声图与独立噪声样本。生成模型不计算分类 accuracy，GAN 判别器分数和像素统计只用于诊断，不当作图像质量分数。

## 输出文件

每个 `runs/<实验名>/` 包含：

| 文件 | 用途 |
|---|---|
| `config.json` | 模型、超参数与 PyTorch 版本 |
| `data_manifest.json` | 数据集、实际样本索引、划分、预处理 |
| `history.csv` | 每轮训练与验证指标，可用 Excel 打开 |
| `best.pt` | 按验证准确率选出的最佳权重及 epoch |
| `test_metrics.json` | 最终测试准确率、macro F1、各类 F1、混淆矩阵 |
| `learning_curves.png` | loss / accuracy 曲线 |
| `confusion_matrix.png` | 测试混淆矩阵 |
| `vocab.json` | 仅文本实验，预测时复用原词表 |

已有 `best.pt` 的运行目录不会被新训练覆盖。改超参数时指定新目录，例如 `runs/resnet_lr001`。

## 离线课堂与排错

- 教师先执行 `prepare_data.py`，然后将生成的 `data/` 目录放到学生项目根目录。训练命令加 `--offline`，不再尝试下载数据。
- 也可手动从官方页下载 `aclImdb_v1.tar.gz`，执行 `python prepare_data.py --dataset imdb --imdb-archive /path/aclImdb_v1.tar.gz`。代码会核验 SHA-1。
- CIFAR-10 缓存应包含 `data/cifar10/cifar-10-batches-py/`；IMDB 缓存应包含 `data/imdb/train.jsonl` 与 `test.jsonl`。预训练扩展另需 PyTorch 权重缓存，基础实验无需预训练权重。
- `torch.cuda.is_available()` 为 False 时先检查内核和 Python 路径。不要仅凭 `nvidia-smi` 判断 PyTorch 可用 CUDA。
- `torchvision::nms` 等导入错误通常意味着 torch / torchvision 安装组合不匹配，按官方安装页重装同一套版本。
- 显存不足先降低 batch size；原生 VGG11 的参数量较大，基础课优先使用默认 CIFAR 适配版。
- 默认 Windows/Jupyter 使用 `workers=0`。命令行多进程已放在 `if __name__ == '__main__'` 保护下。

教师授课安排见 `docs/TEACHER_GUIDE.md`，实验报告模板见 `docs/REPORT_TEMPLATE.md`，参考来源见 `docs/SOURCES.md`。
