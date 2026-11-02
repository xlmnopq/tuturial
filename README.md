# tuturial

个人学习仓库：把「CNN 入门 → D2L 完整实验 → 生成模型课件」串成一条线，本地 Windows 与 GPU 服务器 `amax` 共用同一份代码。

## 目录

| 路径 | 内容 | 入口文档 |
|---|---|---|
| `cnn_quickstart/` | 入门最小集：纯 Python 手写卷积/池化，再用 PyTorch 训一个最小 CNN（Fashion-MNIST 10 分类） | `cnn_quickstart/README.md` |
| `tuturial/d2l_pytorch_labs/` | 完整课程实验：9 个 notebook（00–08），覆盖 CIFAR-10 图像分类（SmallCNN / AlexNet / VGG11 / ResNet18）、IMDB 情感分类（BiLSTM / TextCNN）、生成模型（VAE / DCGAN），含固定数据划分、批量训练与评测脚本 | `tuturial/d2l_pytorch_labs/README.md` |
| `tuturial/vae_gan_build/` | 「VAE 与 GAN 原理对比」课件的构建脚本：读 `content.ndjson`，生成并校验 pptx | `build.mjs` |

`cnn_quickstart` 与 `d2l_pytorch_labs` 是配套关系，不是替代关系：前者是入门最小集（一个 CNN 从建到训），后者是正式实验项目。

环境、工具链与三处同步的完整配置步骤见 [`docs/环境配置与同步手册.md`](docs/环境配置与同步手册.md)。

## 环境

- **本地**：Windows + Git 2.56（`D:\APPstore\Git`）+ VS Code（GitLens）；显卡 GTX 1060 3GB，显存小，batch size 建议 ≤ 64。
- **服务器**：`amax`（Ubuntu 22.04，80 核 / 440 GB 内存，git 2.34），数据盘 `/data1`（7.6 T 可用）。

## 同步方式

同一份代码有三个位置：

```
本地 D:\program\tuturial ──push──> GitHub  xlmnopq/tuturial
        └──push──> amax:/data1/yxy/projects/tuturial.git（裸仓库）──pull──> 服务器工作副本
```

服务器访问不了 github.com（连接超时），所以它只与局域网内的裸仓库同步；GitHub 是本地这一侧的备份。

**本地改完**

```powershell
git add -A
git commit -m "..."
git sync      # 全局别名 = git push origin main && git push amax main
```

**服务器取最新**

```bash
cd /data1/yxy/projects/tuturial && git pull
```

**在服务器上改完**

```bash
git commit -am "..."
git push        # 推到裸仓库
```

本地再 `git pull amax`（不是 `git pull`，那走 GitHub），然后 `git sync` 让三处一致。

## 常用命令

```powershell
# CNN 入门：先看公式，再跑训练
python cnn_quickstart/01_conv_math.py
python cnn_quickstart/02_pytorch_cnn.py

# D2L 实验（步骤与参数见该目录 README）
cd tuturial/d2l_pytorch_labs
pip install -r requirements.txt
python check_environment.py
```

## 版本库约定

- 构建中间产物、渲染图、交付二进制、模型权重不进仓库（规则见根目录 `.gitignore`），由脚本重建；
- 课件的 `authored.json`（约 76 MB）、`template.json`（约 32 MB）是中间态，排除后仓库只有几 MB，克隆和推送都很快；
- 服务器 `/home` 所在分区接近写满，数据集与训练产物请放 `/data1/yxy` 下；
- `tuturial/vae_gan_build/build.mjs` 的 import 里写死了原机器的绝对路径（`C:/Users/jhg/...`），换机器构建前需要改成自己机器上的路径。
