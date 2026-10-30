# 验证范围

验证日期：2026-10-21。

## 实际运行环境

- Windows，NVIDIA GeForce GTX 1650，4 GB 显存。
- 已有 Python 3.8.5、PyTorch 1.12.1+cu113、torchvision 0.13.1+cu113。
- 本地测试使用 `LAB_NUM_THREADS=2`、`workers=0`，图像/文本/生成模型命令行检查 batch size 16。
- Notebook 使用各自配置的 batch size，在 `LAB_SMOKE=1` 下执行。

没有更改已有全局 PyTorch 环境。README 中 Python 3.10/3.11 + PyTorch 2.5.1 / torchvision 0.20.1 是按官方历史版本页提供的新环境安装示例，并非在本机重新安装后实测的版本声明。实现为新旧 API 差异保留了 GradScaler、torch.load 的兼容处理。

## 已完成的检查

| 检查 | 结果 |
|---|---|
| Python 源文件语法解析 | 通过 |
| 9 个 Notebook 的 nbformat 与代码语法 | 通过 |
| 数据划分、分层子集、训练词表隔离、填充处理、指标统计、checkpoint 恢复等 10 项核心单元测试 | 通过 |
| VAE 形状/重参数化梯度、标准正态 KL、GAN 两网络更新、生成模型恢复等 4 项单元测试 | 通过 |
| 4 个图像分类模型的真实 CIFAR-10 GPU 小规模训练与测试 | 通过 |
| BiLSTM / TextCNN 的真实 IMDB GPU 小规模训练与测试 | 通过 |
| VAE / DCGAN 的真实 CIFAR-10 GPU 小规模训练、验证、测试、保存与独立生成 | 通过 |
| 分类、文本、生成模型的独立进程复测 / 预测与结果汇总 | 通过 |
| 9 个 Notebook 的全部代码单元真实数据小规模执行 | 通过 |
| torchvision 原生 AlexNet / VGG11 / ResNet18 的 224×224 前向 | 通过，随机权重，不下载预训练权重 |

核心单元测试包含 CPU 运算与 GPU 模型检查，生成模型单元测试使用微型合成输入；合成输入仅用于软件测试，不是新增的课程数据集。所有教学训练仍只使用 CIFAR-10 和 IMDB。

数据源完整性检查：CIFAR-10 原始压缩包通过官方 MD5，IMDB 原始压缩包通过 D2L 登记的 SHA-1。实际下载并验证了完整原始数据包，训练检查使用其中的分层小样本。

## 验证的实际含义

小规模检查统一使用 128 个训练、64 个验证、64 个测试样本，训练 1 轮。它验证数据加载、梯度更新、权重保存、恢复、测试与生成能连贯运行，**不证明模型已经收敛，也不提供可用来排名的正式准确率或生成质量结论**。

未在本机执行完整数据的 20/30/50 轮收敛训练，也未验证在线下载 ImageNet 权重后的完整迁移学习训练。Notebook 已逐格执行标准 Python 代码，但没有声称在所有 Jupyter 前端和所有 GPU 型号上做过交互测试。

本机的旧 CUDA 库在部分运行结束时输出过非致命的 cuBLAS 符号提示，进程仍成功结束并生成结果。新课堂环境应按照官方安装页统一 torch / torchvision / CUDA wheel 组合。

## 复现验证

```bash
python -m unittest discover -s tests -v
python tools/check_notebooks.py
python tools/smoke_all.py
python tools/run_notebook_smoke.py
```

独立生成模型包仅包含生成模型单元测试和相关 Notebook，使用：

```bash
python -m unittest discover -s tests -p test_generative.py -v
python tools/check_notebooks.py
```

代码包不附带大模型权重、数据缓存和本机私有日志。Notebook 执行清单保存在完整包的 `docs/notebook_execution.json`。
