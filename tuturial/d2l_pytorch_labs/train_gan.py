"""CIFAR-10 DCGAN：交替训练、留出集诊断、保存与随机生成。"""
import argparse
from pathlib import Path

import torch

from labkit.common import choose_device, save_json, seed_everything
from labkit.data import load_image_data
from labkit.generative import fit_gan, gan_diagnostics, noise, plot_generative_history, save_grid
from labkit.generative_models import DCDiscriminator, DCGenerator, dcgan_init


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.0002)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if min(args.batch_size, args.epochs, args.latent_dim, args.channels) < 1 or args.lr <= 0:
        p.error("批量、轮数、潜维度、通道数和学习率必须为正")
    seed_everything(args.seed)
    device = choose_device(args.device)
    limits = [128, 64, 64] if args.smoke else [0, 0, 0]
    bundle = load_image_data(args.data_dir, args.batch_size, args.seed, args.workers,
                             augment=False, download=not args.offline,
                             max_train=limits[0], max_val=limits[1], max_test=limits[2])
    config = dict(vars(args), task="gan", model="dcgan32", epochs=1 if args.smoke else args.epochs,
                  torch_version=str(torch.__version__), input_range=[-1, 1],
                  checkpoint_policy="last epoch; no ranking by discriminator loss")
    generator, discriminator = DCGenerator(args.latent_dim, args.channels), DCDiscriminator(args.channels)
    generator.apply(dcgan_init)
    discriminator.apply(dcgan_init)
    print("device:", device, "data:", bundle.meta["sizes"], flush=True)
    history = fit_gan(generator, discriminator, bundle, device, args.run_dir, config)
    metrics = gan_diagnostics(generator, discriminator, bundle.test, device,
                               args.latent_dim, seed=args.seed + 2000)
    save_json(Path(args.run_dir) / "test_metrics.json", metrics)
    generator.eval()
    with torch.no_grad():
        save_grid(generator(noise(64, args.latent_dim, device, args.seed + 3000)),
                   Path(args.run_dir) / "test_samples.png")
    fig = plot_generative_history(history, "gan", Path(args.run_dir) / "learning_curves.png")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print("GAN test diagnostics:", metrics, flush=True)
    print("D scores/loss and pixel statistics are not image-quality metrics.", flush=True)


if __name__ == "__main__":
    main()
