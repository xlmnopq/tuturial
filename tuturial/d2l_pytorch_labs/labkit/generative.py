"""VAE / GAN 的训练、独立验证、测试与样本保存。"""
from __future__ import annotations

import math
import time
from pathlib import Path

import torch
from torch import nn
from torchvision.utils import save_image

from .common import load_checkpoint, save_json, write_csv
from .generative_models import vae_loss


def save_grid(images, path, nrow=8):
    """模型空间 [-1,1] 映射到显示空间 [0,1]，不重新按每张图拉伸对比度。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(((images.detach().cpu() + 1) / 2).clamp(0, 1), str(path), nrow=nrow)


def cpu_state(model):
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def noise(count, latent_dim, device, seed):
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(count, latent_dim, device=device, generator=generator)


def vae_epoch(model, loader, device, optimizer=None, beta=1.0,
              observation_std=0.1, eval_seed=12345):
    training = optimizer is not None
    model.train(training)
    # 独立随机数生成器不改变下一轮训练的随机状态。
    generator = None if training else torch.Generator(device=device).manual_seed(eval_seed)
    totals = {"objective": 0.0, "reconstruction_nll": 0.0, "kl": 0.0,
              "negative_elbo": 0.0, "reconstruction_mse01": 0.0}
    count = 0
    start = time.perf_counter()
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            reconstruction, mu, logvar = model(images, sample=True, generator=generator)
            objective, nll, kl = vae_loss(reconstruction, images, mu, logvar, beta, observation_std)
            if not torch.isfinite(objective):
                raise FloatingPointError("VAE loss 出现 NaN/Inf，请降低学习率并检查方差")
            if training:
                objective.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            # 训练时记录采样重构，验证/测试时记录 decode(mu) 的确定性重构。
            with torch.no_grad():
                display_reconstruction = reconstruction if training else model.decode(mu)
                mse01 = ((display_reconstruction - images) / 2).square().mean()
        n = images.size(0)
        values = {"objective": objective.item(), "reconstruction_nll": nll.item(),
                  "kl": kl.item(), "negative_elbo": nll.item() + kl.item(),
                  "reconstruction_mse01": mse01.item()}
        for key, value in values.items():
            totals[key] += value * n
        count += n
    if not count:
        raise ValueError("DataLoader 为空")
    metrics = {key: value / count for key, value in totals.items()}
    metrics.update(samples=count, seconds=time.perf_counter() - start)
    metrics["reconstruction_psnr_db"] = -10 * math.log10(max(metrics["reconstruction_mse01"], 1e-12))
    return metrics


@torch.no_grad()
def save_vae_examples(model, loader, device, run_dir, stem="test", seed=2026, count=16):
    model.eval()
    images = next(iter(loader))[0][:count].to(device)
    reconstruction, _, _ = model(images, sample=False)
    run_dir = Path(run_dir)
    save_grid(torch.cat([images, reconstruction]), run_dir / (stem + "_reconstruction.png"), nrow=len(images))
    generated = model.decode(noise(64, model.latent_dim, device, seed))
    save_grid(generated, run_dir / (stem + "_samples.png"))


def fit_vae(model, bundle, device, run_dir, config):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "best.pt").exists():
        raise FileExistsError("运行目录已有 best.pt，请使用新的运行目录")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    save_json(run_dir / "config.json", config)
    save_json(run_dir / "data_manifest.json", bundle.meta)
    history, best = [], float("inf")
    for epoch in range(1, config["epochs"] + 1):
        train = vae_epoch(model, bundle.train, device, optimizer, config["beta"], config["observation_std"])
        val = vae_epoch(model, bundle.val, device, beta=config["beta"], observation_std=config["observation_std"])
        row = {"epoch": epoch}
        row.update({"train_" + k: v for k, v in train.items()})
        row.update({"val_" + k: v for k, v in val.items()})
        history.append(row)
        write_csv(run_dir / "history.csv", history)
        print("VAE epoch %d/%d train=%.3f val=%.3f KL=%.3f" %
              (epoch, config["epochs"], train["objective"], val["objective"], val["kl"]), flush=True)
        if val["objective"] < best:
            best = val["objective"]
            torch.save({"model_state": cpu_state(model), "config": config,
                        "best_epoch": epoch, "best_val_objective": best}, run_dir / "best.pt")
        save_vae_examples(model, bundle.val, device, run_dir / "previews",
                           "epoch_%03d" % epoch, seed=config["seed"] + 1000)
    model.load_state_dict(load_checkpoint(run_dir / "best.pt")["model_state"])
    return history


def gan_train_step(generator, discriminator, real, optimizer_g, optimizer_d, latent_dim):
    """先更新 D，再更新 G。D 看见的 fake.detach() 不会把梯度传到 G。"""
    generator.train()
    discriminator.train()
    for parameter in discriminator.parameters():
        parameter.requires_grad_(True)
    batch_size = real.size(0)
    z = torch.randn(batch_size, latent_dim, device=real.device)
    fake = generator(z)
    criterion = nn.BCEWithLogitsLoss()

    # 1. 更新判别器：真图为 1，生成图为 0。
    optimizer_d.zero_grad(set_to_none=True)
    logits_real = discriminator(real)
    logits_fake = discriminator(fake.detach())
    loss_d = 0.5 * (criterion(logits_real, torch.ones_like(logits_real)) +
                    criterion(logits_fake, torch.zeros_like(logits_fake)))
    if not torch.isfinite(loss_d):
        raise FloatingPointError("GAN 判别器损失出现 NaN/Inf")
    loss_d.backward()
    optimizer_d.step()
    optimizer_d.zero_grad(set_to_none=True)

    # 2. 更新生成器：冻结 D 参数与 BN 统计量，但保留 D 对输入的梯度。
    for parameter in discriminator.parameters():
        parameter.requires_grad_(False)
    discriminator.eval()
    optimizer_g.zero_grad(set_to_none=True)
    logits_for_g = discriminator(fake)
    loss_g = criterion(logits_for_g, torch.ones_like(logits_for_g))
    if not torch.isfinite(loss_g):
        raise FloatingPointError("GAN 生成器损失出现 NaN/Inf")
    loss_g.backward()
    optimizer_g.step()
    for parameter in discriminator.parameters():
        parameter.requires_grad_(True)
    discriminator.train()
    return {"d_loss": loss_d.item(), "g_loss": loss_g.item(),
            "d_real": logits_real.detach().sigmoid().mean().item(),
            "d_fake": logits_fake.detach().sigmoid().mean().item()}


@torch.no_grad()
def gan_diagnostics(generator, discriminator, loader, device, latent_dim, seed=12345):
    """留出集诊断，不是图像质量评分，也不能作为 G 的最佳模型选择依据。"""
    generator.eval()
    discriminator.eval()
    rng = torch.Generator(device=device).manual_seed(seed)
    totals = {"d_loss_diagnostic": 0.0, "d_real_score": 0.0, "d_fake_score": 0.0}
    sum_real = torch.zeros(3, device=device, dtype=torch.float64)
    sum_fake = sum_real.clone()
    square_real, square_fake = sum_real.clone(), sum_real.clone()
    count, pixels = 0, 0
    criterion = nn.BCEWithLogitsLoss()
    for images, _ in loader:
        images = images.to(device)
        size = images.size(0)
        z = torch.randn(size, latent_dim, device=device, generator=rng)
        fake = generator(z)
        real_logits, fake_logits = discriminator(images), discriminator(fake)
        d_loss = 0.5 * (criterion(real_logits, torch.ones_like(real_logits)) +
                        criterion(fake_logits, torch.zeros_like(fake_logits)))
        totals["d_loss_diagnostic"] += d_loss.item() * size
        totals["d_real_score"] += real_logits.sigmoid().sum().item()
        totals["d_fake_score"] += fake_logits.sigmoid().sum().item()
        real01, fake01 = (images.double() + 1) / 2, (fake.double() + 1) / 2
        sum_real += real01.sum((0, 2, 3))
        sum_fake += fake01.sum((0, 2, 3))
        square_real += real01.square().sum((0, 2, 3))
        square_fake += fake01.square().sum((0, 2, 3))
        pixels += size * images.size(2) * images.size(3)
        count += size
    if count == 0:
        raise ValueError("DataLoader 为空")
    result = {key: value / count for key, value in totals.items()}
    for kind, sums, squares in [("real", sum_real, square_real), ("generated", sum_fake, square_fake)]:
        mean = sums / pixels
        result[kind + "_channel_mean01"] = mean.cpu().tolist()
        result[kind + "_channel_std01"] = (squares / pixels - mean.square()).clamp(min=0).sqrt().cpu().tolist()
    result.update(samples=count, scope="full" if count == 10000 else "subset",
                  interpretation="Discriminator scores and pixel statistics are diagnostics, not generation quality metrics.")
    return result


def fit_gan(generator, discriminator, bundle, device, run_dir, config):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "last.pt").exists():
        raise FileExistsError("运行目录已有 last.pt，请使用新的运行目录")
    generator.to(device)
    discriminator.to(device)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=config["lr"], betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=config["lr"], betas=(0.5, 0.999))
    save_json(run_dir / "config.json", config)
    save_json(run_dir / "data_manifest.json", bundle.meta)
    fixed_noise = noise(64, config["latent_dim"], device, config["seed"] + 1000)
    history = []
    for epoch in range(1, config["epochs"] + 1):
        totals = {"d_loss": 0.0, "g_loss": 0.0, "d_real": 0.0, "d_fake": 0.0}
        count = 0
        start = time.perf_counter()
        for real, _ in bundle.train:
            real = real.to(device)
            row = gan_train_step(generator, discriminator, real, optimizer_g, optimizer_d, config["latent_dim"])
            for key, value in row.items():
                totals[key] += value * real.size(0)
            count += real.size(0)
        val = gan_diagnostics(generator, discriminator, bundle.val, device, config["latent_dim"])
        row = dict(epoch=epoch, train_seconds=time.perf_counter() - start,
                   **{key: value / count for key, value in totals.items()})
        row.update({"val_" + key: val[key] for key in ["d_loss_diagnostic", "d_real_score", "d_fake_score"]})
        history.append(row)
        write_csv(run_dir / "history.csv", history)
        print("GAN epoch %d/%d D=%.4f G=%.4f D(real)=%.3f D(fake)=%.3f" %
              (epoch, config["epochs"], row["d_loss"], row["g_loss"], row["d_real"], row["d_fake"]), flush=True)
        generator.eval()
        with torch.no_grad():
            save_grid(generator(fixed_noise), run_dir / "previews" / ("epoch_%03d.png" % epoch))
        # 训练预算由实验配置预先确定，不按判别器 loss 伪造一个“最佳生成器”。
        torch.save({"generator_state": cpu_state(generator),
                    "discriminator_state": cpu_state(discriminator),
                    "epoch": epoch, "config": config}, run_dir / "last.pt")
    return history


def plot_generative_history(history, kind, path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = [r["epoch"] for r in history]
    if kind == "vae":
        for split in ["train", "val"]:
            axes[0].plot(epochs, [r[split + "_objective"] for r in history], label=split)
            axes[1].plot(epochs, [r[split + "_kl"] for r in history], label=split)
        axes[0].set_ylabel("Objective per image (nats)")
        axes[1].set_ylabel("KL per image (nats)")
    else:
        for key in ["d_loss", "g_loss"]:
            axes[0].plot(epochs, [r[key] for r in history], label=key)
        for key in ["d_real", "d_fake"]:
            axes[1].plot(epochs, [r[key] for r in history], label=key)
        axes[0].set_ylabel("Training loss (diagnostic)")
        axes[1].set_ylabel("Discriminator score (diagnostic)")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    return fig
