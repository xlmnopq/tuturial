"""本地验证用的可续传分块下载，最后按原始数据集校验和检查。

学生通常直接使用 prepare_data.py；此工具用于网络较慢时预备课堂缓存。
"""
import hashlib
import json
import urllib.request
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).absolute().parents[1]


def ranged_download(url, target, expected, algorithm="md5", workers=8):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    def digest(path):
        h = hashlib.new(algorithm)
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    if target.exists() and digest(target) == expected:
        print("Already verified", target.name, flush=True)
        return
    with requests.get(url, headers={"Range": "bytes=0-0"}, stream=True, timeout=(20, 45)) as response:
        if response.status_code != 206:
            raise RuntimeError("服务器不支持 Range，请改用 prepare_data.py")
        total = int(response.headers["Content-Range"].split("/")[-1])
    folder = target.parent / (target.name + ".chunks")
    folder.mkdir(exist_ok=True)
    chunk_size = 4 * 1024**2
    ranges = [(i, start, min(total - 1, start + chunk_size - 1))
              for i, start in enumerate(range(0, total, chunk_size))]
    def fetch(job):
        i, start, end = job
        path = folder / ("part_%03d" % i)
        if path.exists() and path.stat().st_size == end - start + 1:
            return path
        for attempt in range(3):
            try:
                with requests.get(url, headers={"Range": "bytes=%d-%d" % (start, end)},
                                   stream=True, timeout=(20, 60)) as response:
                    actual = response.headers.get("Content-Range", "")
                    if response.status_code != 206 or not actual.startswith("bytes %d-%d/" % (start, end)):
                        raise RuntimeError("服务器返回的分段范围不匹配")
                    with path.open("wb") as f:
                        for data in response.iter_content(256 * 1024):
                            if not data:
                                continue
                            f.write(data)
                if path.stat().st_size != end - start + 1:
                    raise RuntimeError("分段长度不匹配")
                return path
            except Exception:
                if attempt == 2:
                    raise
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, job) for job in ranges]
        for done, future in enumerate(as_completed(futures), 1):
            print(target.name, done, "/", len(ranges), future.result().name, flush=True)
    assembled = target.with_suffix(target.suffix + ".assembled")
    with assembled.open("wb") as out:
        for i, _, _ in ranges:
            with (folder / ("part_%03d" % i)).open("rb") as part:
                for data in iter(lambda: part.read(1024 * 1024), b""):
                    out.write(data)
    if digest(assembled) != expected:
        raise RuntimeError("完整文件校验失败")
    assembled.replace(target)
    print("Checksum verified:", target, flush=True)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT))
    from prepare_data import CIFAR_MIRROR, main
    from labkit.data import IMDB_URL, IMDB_SHA1
    ranged_download(CIFAR_MIRROR, ROOT / "data/cifar10/cifar-10-python.tar.gz",
                     "c58f30108f718f92721af3b95e74349a")
    # IMDB 官方服务器支持分段时可续传，不支持时保留标准下载路径。
    try:
        ranged_download(IMDB_URL, ROOT / "data/imdb/aclImdb_v1.tar.gz", IMDB_SHA1, "sha1")
    except RuntimeError as error:
        print("IMDB range download:", error, flush=True)
    sys.argv = [sys.argv[0], "--dataset", "all"]
    main()
