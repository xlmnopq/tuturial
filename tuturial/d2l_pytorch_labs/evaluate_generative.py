"""独立加载保存的生成模型，并按原运行的测试索引复测。"""
import argparse
import json
from pathlib import Path

from torch.utils.data import Subset
from torchvision.datasets import CIFAR10

from labkit.common import save_json
from labkit.data import image_transforms, make_loader
from labkit.generative import gan_diagnostics, save_vae_examples, vae_epoch
from labkit.generative_inference import restore_generative


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", default="auto")
    p.add_argument("--full-test", action="store_true")
    args = p.parse_args()
    model, discriminator, config, device = restore_generative(args.run_dir, args.device)
    manifest = json.loads((Path(args.run_dir) / "data_manifest.json").read_text(encoding="utf-8"))
    _, transform, _, _ = image_transforms("cifar", augment=False)
    source = CIFAR10(str(Path(args.data_dir) / "cifar10"), train=False, download=False, transform=transform)
    subset = source if args.full_test else Subset(source, manifest["test_indices"])
    loader = make_loader(subset, args.batch_size, False, config["seed"])
    if config["task"] == "vae":
        metrics = vae_epoch(model, loader, device, beta=config["beta"], observation_std=config["observation_std"])
        metrics.update(loss_units="nats per image", mc_samples_per_image=1,
                       reconstruction_metric="decode(mu), MSE averaged over pixels in [0,1]")
        save_vae_examples(model, loader, device, args.run_dir, "reevaluation", config["seed"] + 2000)
    else:
        metrics = gan_diagnostics(model, discriminator, loader, device, config["latent_dim"], config["seed"] + 2000)
    save_json(Path(args.run_dir) / "reevaluation.json", metrics)
    print(metrics)


if __name__ == "__main__":
    main()
