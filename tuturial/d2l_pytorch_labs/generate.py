"""从保存的 VAE 或 GAN 生成图像，不需要重新训练或加载数据集。"""
import argparse
from pathlib import Path

import torch

from labkit.generative import noise, save_grid
from labkit.generative_inference import restore_generative


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--num-samples", type=int, default=64)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--output", default=None)
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    if not 1 <= args.num_samples <= 1024:
        p.error("num-samples 必须在 1 到 1024 之间")
    model, _, config, device = restore_generative(args.run_dir, args.device)
    output = Path(args.output) if args.output else Path(args.run_dir) / ("generated_seed%d.png" % args.seed)
    if output.exists():
        raise FileExistsError("输出图已存在，请指定新的 --output")
    chunks = []
    # 一次最多生成 64 张，避免大网格请求造成 GPU 显存突增。
    z = noise(args.num_samples, config["latent_dim"], device, args.seed)
    with torch.no_grad():
        for piece in z.split(64):
            images = model.decode(piece) if config["task"] == "vae" else model(piece)
            chunks.append(images.cpu())
    save_grid(torch.cat(chunks), output)
    print("Generated:", output)
    if config["task"] == "vae":
        print("显示解码器的条件均值，不额外叠加高斯观测噪声。")


if __name__ == "__main__":
    main()
