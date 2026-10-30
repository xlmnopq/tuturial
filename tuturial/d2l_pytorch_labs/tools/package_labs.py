"""生成完整课程包和独立 VAE/GAN 实操包，不包含数据、权重或私有运行日志。"""
from pathlib import Path
import zipfile

ROOT = Path(__file__).absolute().parents[1]
OUT = ROOT.parent / "output"
OUT.mkdir(exist_ok=True)


def allowed(path):
    rel = path.relative_to(ROOT)
    if any(part in {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"} for part in rel.parts):
        return False
    if path.name.endswith("_run_path.txt") or path.name == "smoke_run_path.txt":
        return False
    if rel.parts[0] in {"data", "runs"}:
        return False
    return path.is_file() and path.suffix in {".py", ".md", ".txt", ".ipynb", ".json"}


def main():
    full = OUT / "PyTorch_学生实操完整包.zip"
    small = OUT / "CIFAR10_VAE_GAN_实操包.zip"
    if full.exists() or small.exists():
        raise FileExistsError("成品压缩包已存在，请为新版本选择新文件名")
    with zipfile.ZipFile(full, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(ROOT.rglob("*")):
            if allowed(path):
                rel = path.relative_to(ROOT)
                archive.write(path, str(Path("d2l_pytorch_labs") / rel))
    keep_root = ["check_environment.py", "prepare_data.py", "train_vae.py", "train_gan.py",
                 "generate.py", "evaluate_generative.py", "requirements.txt"]
    keep_modules = ["__init__.py", "common.py", "data.py", "generative_models.py",
                    "generative.py", "generative_inference.py"]
    standalone_intro = """# CIFAR-10 VAE / GAN 学生实操包

本包包含卷积 VAE、DCGAN 两个实验的脚本和逐格 Notebook，支持 NVIDIA GPU。

## 安装与开始

建议使用独立的 Python 3.10 / 3.11 环境，先进入本 README 所在目录。

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\Activate.ps1
# Linux: source .venv/bin/activate
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python check_environment.py
python prepare_data.py --dataset cifar10 --cifar-mirror
python train_vae.py --smoke --run-dir runs/vae_check
python train_gan.py --smoke --run-dir runs/gan_check
```

上述固定版本是课堂安装示例，不是最新版本声明。新 GPU 架构或已有环境请按 PyTorch 官方安装选择器选择匹配组合。实际验证环境与范围见 `docs/VALIDATION.md`。

使用 Notebook 时执行 `python -m ipykernel install --user --name d2l-labs --display-name "PyTorch Labs"`，然后 `python -m jupyterlab`。先运行 00 环境检查，再选择 07 VAE 或 08 GAN。`SMOKE=True` 只验证流程。

数据不包含在压缩包内。教师下载一次后可分发 `data/` 缓存，学生添加 `--offline`。本项目不需要第三个数据集或额外预训练网络。

以下为完整实验说明。

"""
    paths = [ROOT / name for name in keep_root]
    paths += [ROOT / "labkit" / name for name in keep_modules]
    paths += [ROOT / "notebooks" / name for name in ["00_environment.ipynb", "07_cifar10_vae.ipynb", "08_cifar10_gan.ipynb"]]
    paths += [ROOT / "docs" / name for name in ["GENERATIVE_GUIDE.md", "VALIDATION.md", "SOURCES.md", "REPORT_TEMPLATE.md"]]
    paths += [ROOT / "tests/test_generative.py", ROOT / "tools/check_notebooks.py"]
    with zipfile.ZipFile(small, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, str(Path("cifar10_generative_labs") / path.relative_to(ROOT)))
        archive.writestr("cifar10_generative_labs/README.md", standalone_intro +
                         (ROOT / "docs/GENERATIVE_GUIDE.md").read_text(encoding="utf-8"))
    for path in [small, full]:
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:
                raise RuntimeError("ZIP integrity failed")
            print(path, "files:", len(archive.namelist()), "bytes:", path.stat().st_size)


if __name__ == "__main__":
    main()
