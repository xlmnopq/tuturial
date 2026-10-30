"""32×32 RGB 的卷积 VAE 与 DCGAN，输入/输出范围统一为 [-1, 1]。"""
import math

import torch
from torch import nn


class ConvVAE(nn.Module):
    def __init__(self, latent_dim=64, channels=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.channels = channels
        self.encoder = nn.Sequential(
            nn.Conv2d(3, channels, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(channels, channels * 2, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(channels * 2, channels * 4, 4, 2, 1), nn.ReLU(),
            nn.Flatten())
        self.mu = nn.Linear(channels * 4 * 4 * 4, latent_dim)
        self.logvar = nn.Linear(channels * 4 * 4 * 4, latent_dim)
        self.project = nn.Linear(latent_dim, channels * 4 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(channels * 4, channels * 2, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(channels * 2, channels, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(channels, 3, 4, 2, 1), nn.Tanh())

    def encode(self, x):
        features = self.encoder(x)
        # 约束极端方差，避免 exp(logvar) 溢出；这是本教学实现的数值选择。
        return self.mu(features), self.logvar(features).clamp(-20, 10)

    def reparameterize(self, mu, logvar, generator=None):
        eps = torch.randn(mu.shape, dtype=mu.dtype, device=mu.device, generator=generator)
        return mu + torch.exp(0.5 * logvar) * eps

    def decode(self, z):
        features = self.project(z).reshape(-1, self.channels * 4, 4, 4)
        return self.decoder(features)

    def forward(self, x, sample=True, generator=None):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, generator) if sample else mu
        return self.decode(z), mu, logvar


def vae_loss(reconstruction, target, mu, logvar, beta=1.0, observation_std=0.1):
    """固定方差高斯像素似然，每张图像按全部像素求和，然后对 batch 求均值。

    beta=1: 负 ELBO 的一次 Monte Carlo 估计；beta!=1: beta-VAE 目标。
    保留高斯常数项，避免把任意缩放的 MSE 与负对数似然混淆。
    """
    if observation_std <= 0 or beta < 0:
        raise ValueError("observation_std 必须为正，beta 不能为负")
    variance = observation_std**2
    reconstruction_nll = 0.5 * (
        (target - reconstruction).square() / variance + math.log(2 * math.pi * variance)
    ).flatten(1).sum(1).mean()
    kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(1).mean()
    objective = reconstruction_nll + beta * kl
    return objective, reconstruction_nll, kl


class DCGenerator(nn.Module):
    def __init__(self, latent_dim=64, channels=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, channels * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(channels * 4), nn.ReLU(),
            nn.ConvTranspose2d(channels * 4, channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels * 2), nn.ReLU(),
            nn.ConvTranspose2d(channels * 2, channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(),
            nn.ConvTranspose2d(channels, 3, 4, 2, 1, bias=False), nn.Tanh())

    def forward(self, z):
        return self.net(z.reshape(-1, self.latent_dim, 1, 1))


class DCDiscriminator(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, channels, 4, 2, 1, bias=False), nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels * 2), nn.LeakyReLU(0.2),
            nn.Conv2d(channels * 2, channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels * 4), nn.LeakyReLU(0.2),
            nn.Conv2d(channels * 4, 1, 4, 1, 0, bias=False))

    def forward(self, x):
        # 输出 logits；BCEWithLogitsLoss 已经包含 sigmoid 的数值稳定实现。
        return self.net(x).flatten()


def dcgan_init(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)
