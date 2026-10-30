"""CIFAR-10 卷积 VAE：数据、训练、验证、测试、重构与随机生成。"""
import argparse
from pathlib import Path

import torch

from labkit.common import choose_device, save_json, seed_everything
from labkit.data import load_image_data
from labkit.generative import fit_vae, plot_generative_history, save_vae_examples, vae_epoch
from labkit.generative_models import ConvVAE


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--observation-std", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if min(args.batch_size, args.epochs, args.latent_dim, args.channels) < 1 or args.lr <= 0:
        p.error("批量、轮数、潜维度、通道数和学习率必须为正")
    if args.beta < 0 or args.observation_std <= 0:
        p.error("beta >= 0，observation-std > 0")
    seed_everything(args.seed)
    device = choose_device(args.device)
    limits = [128, 64, 64] if args.smoke else [0, 0, 0]
    bundle = load_image_data(args.data_dir, args.batch_size, args.seed, args.workers,
                             augment=False, download=not args.offline,
                             max_train=limits[0], max_val=limits[1], max_test=limits[2])
    config = dict(vars(args), task="vae", model="conv_vae", epochs=1 if args.smoke else args.epochs,
                  torch_version=str(torch.__version__), input_range=[-1, 1],
                  observation_model="Gaussian with fixed standard deviation in [-1,1] space")
    model = ConvVAE(args.latent_dim, args.channels)
    print("device:", device, "data:", bundle.meta["sizes"], flush=True)
    history = fit_vae(model, bundle, device, args.run_dir, config)
    # 用最佳验证权重在独立官方测试集评估，测试标签不参与 VAE 训练。
    metrics = vae_epoch(model, bundle.test, device, beta=args.beta, observation_std=args.observation_std)
    metrics.update(test_scope="subset" if args.smoke else "full", mc_samples_per_image=1,
                   reconstruction_metric="decode(mu); MSE averaged over all pixels in [0,1] space",
                   loss_units="nats per image", beta=args.beta, observation_std=args.observation_std)
    save_json(Path(args.run_dir) / "test_metrics.json", metrics)
    save_vae_examples(model, bundle.test, device, args.run_dir, "test", seed=args.seed + 2000)
    fig = plot_generative_history(history, "vae", Path(args.run_dir) / "learning_curves.png")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print("VAE test:", metrics, flush=True)


if __name__ == "__main__":
    main()
