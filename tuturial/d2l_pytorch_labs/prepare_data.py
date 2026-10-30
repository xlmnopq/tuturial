"""教师可课前一次性下载两个数据集，随后全班复用数据目录。"""
import argparse
import hashlib
import urllib.request
from pathlib import Path

from torchvision.datasets import CIFAR10
from labkit.data import download_imdb

CIFAR_MIRROR = ("https://huggingface.co/datasets/MIT-OL-AI-D/cifar-10-python/resolve/"
                "a48007227f9e2cd0af96175f4afb5ac4965e261b/cifar-10-python.tar.gz")


def prepare_mirror(root):
    """可选高校镜像；数据仍必须通过官方 MD5 检查。"""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    target = root / "cifar-10-python.tar.gz"
    def valid(path):
        if not path.exists():
            return False
        digest = hashlib.md5()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest() == "c58f30108f718f92721af3b95e74349a"
    if valid(target):
        return
    partial = root / "cifar-10-python.tar.gz.part"
    print("CIFAR-10 optional mirror:", CIFAR_MIRROR, flush=True)
    with urllib.request.urlopen(CIFAR_MIRROR, timeout=60) as response, partial.open("wb") as f:
        total = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            total += len(chunk)
            if total % (20 * 1024 * 1024) == 0:
                print("Downloaded %d MB" % (total // 1024**2), flush=True)
    if not valid(partial):
        raise RuntimeError("CIFAR-10 mirror MD5 mismatch")
    partial.replace(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["all", "cifar10", "imdb"], default="all")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--imdb-archive", help="已离线下载的官方 aclImdb_v1.tar.gz")
    parser.add_argument("--cifar-mirror", action="store_true", help="使用固定版本的高校镜像，仍核对官方 MD5")
    args = parser.parse_args()
    if args.dataset in {"all", "cifar10"}:
        root = str(Path(args.data_dir) / "cifar10")
        if args.cifar_mirror:
            prepare_mirror(root)
        CIFAR10(root, train=True, download=True)
        CIFAR10(root, train=False, download=False)
        print("CIFAR-10 ready", flush=True)
    if args.dataset in {"all", "imdb"}:
        download_imdb(args.data_dir, archive=args.imdb_archive)
        print("IMDB ready", flush=True)


if __name__ == "__main__":
    main()
