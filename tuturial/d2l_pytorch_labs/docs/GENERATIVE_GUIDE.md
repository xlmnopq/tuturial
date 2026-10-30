# CIFAR-10 上的 VAE 与 GAN

两种模型都复用原来图像分类的 CIFAR-10 原始数据与 `seed=42` 数据划分：45,000 训练 / 5,000 验证 / 10,000 测试。生成训练不使用类别标签，也不添加随机图像增广。输入范围统一为 [-1,1]，模型输出通过 Tanh 限制到同一范围。

## 快速跑通

环境与数据准备沿用根目录 README。

```bash
python prepare_data.py --dataset cifar10
python train_vae.py --smoke --run-dir runs/vae_check
python train_gan.py --smoke --run-dir runs/gan_check
```

`--smoke` 只用 128/64/64 个真实样本，训练 1 轮，检查完整管线。它不能生成有代表性的模型质量结果。

## 正式训练

```bash
python train_vae.py --epochs 30 --batch-size 128 --latent-dim 64 --run-dir runs/vae
python train_gan.py --epochs 50 --batch-size 128 --latent-dim 64 --run-dir runs/gan
```

有完整本地缓存时加 `--offline`，在指定 GPU 上运行可加 `--device cuda:0`。显存不足时降低 batch size，例如 32。默认配置是教学起点，未承诺特定重构质量或生成质量。

两个对应 Notebook 为 `07_cifar10_vae.ipynb`、`08_cifar10_gan.ipynb`，包含源码展示、完整运行单元格、图像查看与练习。

## VAE 的具体设置

- 编码器将 32×32 下采样到 4×4，分别输出 `mu` 和 `logvar`，默认潜变量 64 维。
- 重参数化得到 `z`，解码器使用转置卷积恢复为 RGB 图像。
- 像素分布设为固定标准差的高斯，默认 `--observation-std 0.1`，对应归一化后的 [-1,1] 空间。
- 重构 NLL 按每张图像的全部通道/像素求和，KL 按潜变量维度求和，再对 batch 求均值。
- 目标为 `reconstruction_nll + beta * KL`。默认 beta=1，对应负 ELBO 的单样本 Monte Carlo 估计。beta≠1 时是 beta-VAE 目标，脚本仍单独记录未加权的负 ELBO 估计。
- 数值稳定处理对 logvar 使用 [-20,10] 截断。训练采用 float32，没有把复杂 AMP 细节混进基础生成实验。

按验证目标最小值保存 `best.pt`。训练结束恢复该模型，测试时报告 NLL、KL、负 ELBO 与重构指标。

`reconstruction_mse01` / PSNR 在验证和测试阶段使用 `decode(mu)`，在显示范围 [0,1] 上计算。PSNR 由全体像素的平均 MSE 得到，不是每张图 PSNR 的简单平均。它评价配对重构，不评价随机生成。训练曲线中的重构 MSE 来自采样潜变量，二者的采样方式不同。

连续密度可以超过 1，因此包含高斯常数项的 NLL 可能为负。跨配置比较时应固定 `observation_std`、beta、图像缩放和损失归约，不能只看一个 loss 数值。

输出包括：

- `history.csv`：每轮训练/验证目标、重构 NLL、KL、负 ELBO、MSE / PSNR。
- `best.pt`：最佳验证权重、配置与 epoch。
- `previews/epoch_XXX_reconstruction.png`：上排原图，下排潜变量均值解码后的图像。
- `previews/epoch_XXX_samples.png`：每轮同一组先验噪声的解码结果。
- `test_metrics.json`、`test_reconstruction.png`、`test_samples.png`。

随机生成图显示的是解码器的条件均值，不额外叠加固定高斯像素观测噪声。

## GAN 的具体设置

采用处理 32×32 RGB 的 DCGAN。生成器将标准正态噪声通过转置卷积生成图像，判别器以卷积下采样输出一个 logit。

每个 batch 的训练步骤：

1. 由 G 生成 fake。更新 D 时对 fake 执行 `detach()`，真实标签为 1，生成标签为 0，真假 BCE 损失取平均。
2. 固定 D 的参数和 BatchNorm 统计量，保留 D 对输入的梯度，使用目标 1 计算非饱和 G 损失。
3. 用两个独立 Adam 优化器更新，默认 lr=0.0002、betas=(0.5,0.999)。

训练时每轮用同一组固定噪声保存网格，便于观察变化。按预先设定的训练轮数保存 `last.pt`，**不按 D loss 保存一个声称质量最好的 G**。

每轮在验证图像上记录诊断，训练完成后对独立测试图像和独立噪声生成图记录 D 分数，以及两者在 [0,1] 空间的 RGB 通道均值/标准差。

这些数值只帮助发现明显失衡和颜色分布问题，既不是分类准确率，也不等于生成语义质量或多样性。本基础实验不下载额外的 FID 特征网络，不把像素统计称为 FID。若需要正式的生成质量比较，应另行固定特征提取器、样本量与 FID/KID 等协议。

输出包括 `last.pt`、`history.csv`、`learning_curves.png`、`previews/epoch_XXX.png`、`test_samples.png` 和 `test_metrics.json`。G 与 D 权重都保存在 last.pt，生成阶段只使用 G。

## 独立测试、恢复与生成

```bash
python evaluate_generative.py --run-dir runs/vae
python evaluate_generative.py --run-dir runs/gan
python generate.py --run-dir runs/vae --num-samples 64 --seed 2026
python generate.py --run-dir runs/gan --num-samples 64 --seed 2026
```

生成只需要运行目录中的模型与配置，不需要重新加载训练数据。相同模型、设备环境和噪声种子可复现同一组图像。默认输出不覆盖已有图片，可通过 `--output 新路径.png` 指定目标。

`evaluate_generative.py` 默认复用原运行保存的测试索引，`--full-test` 才强制使用完整官方测试集。VAE 测试 NLL 是一次 Monte Carlo 估计，改变 batch size 会改变噪声分组，可能使该估计有轻微变化；确定性 `decode(mu)` 重构不受该采样影响。

## 练习

1. VAE：固定数据与预算，比较 latent_dim=16、64、128。
2. VAE：比较 beta=0.1、1、4，同时报告 KL 和重构项，观察潜变量利用情况。
3. GAN：单独调整 G 或 D 学习率，观察失衡与模式坍塌。
4. GAN：观察训练期固定噪声与最终独立噪声样本，记录重复、颜色和可辨识结构。
5. 说明为何 VAE 的重构 PSNR 与 GAN 的判别器 loss 不能直接放在一个表中当作生成质量排名。

所有图片均由本地实际训练/推理生成。课堂中出现模糊、重复、噪声图也是需要解释的实验现象。
