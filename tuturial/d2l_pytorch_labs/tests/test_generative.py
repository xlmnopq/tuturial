import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from labkit.common import seed_everything
from labkit.data import DataBundle
from labkit.generative import (fit_gan, fit_vae, gan_diagnostics, gan_train_step,
                               noise, vae_epoch)
from labkit.generative_inference import restore_generative
from labkit.generative_models import ConvVAE, DCDiscriminator, DCGenerator, vae_loss


class GenerativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_vae_shapes_and_gradients(self):
        seed_everything(42)
        model = ConvVAE(latent_dim=8, channels=8)
        x = torch.rand(2, 3, 32, 32) * 2 - 1
        reconstruction, mu, logvar = model(x)
        self.assertEqual(reconstruction.shape, x.shape)
        self.assertEqual(mu.shape, (2, 8))
        loss, nll, kl = vae_loss(reconstruction, x, mu, logvar)
        self.assertTrue(torch.allclose(loss, nll + kl))
        self.assertGreaterEqual(kl.item(), -1e-6)
        loss.backward()
        self.assertGreater(model.mu.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.logvar.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.decoder[0].weight.grad.abs().sum().item(), 0)

    def test_standard_normal_kl_is_zero(self):
        x = torch.zeros(2, 3, 32, 32)
        _, _, kl = vae_loss(x, x, torch.zeros(2, 4), torch.zeros(2, 4))
        self.assertEqual(kl.item(), 0)

    def test_gan_step_updates_both_networks(self):
        seed_everything(42)
        g, d = DCGenerator(8, 8), DCDiscriminator(8)
        before_g = next(g.parameters()).detach().clone()
        before_d = next(d.parameters()).detach().clone()
        real = torch.rand(4, 3, 32, 32) * 2 - 1
        values = gan_train_step(g, d, real, torch.optim.Adam(g.parameters()),
                                torch.optim.Adam(d.parameters()), latent_dim=8)
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in values.values()))
        self.assertFalse(torch.equal(before_g, next(g.parameters())))
        self.assertFalse(torch.equal(before_d, next(d.parameters())))
        # D 更新后清空梯度，G 更新不重新积累 D 参数梯度。
        self.assertTrue(all(p.grad is None for p in d.parameters()))

    def test_vae_and_gan_roundtrip(self):
        seed_everything(42)
        images = torch.rand(4, 3, 32, 32) * 2 - 1
        loader = DataLoader(TensorDataset(images, torch.zeros(4, dtype=torch.long)), batch_size=2)
        bundle = DataBundle(loader, loader, loader, {"dataset": "cifar10"})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vae = ConvVAE(8, 8)
            config = {"task": "vae", "epochs": 1, "lr": 0.001, "latent_dim": 8,
                      "channels": 8, "seed": 42, "beta": 1.0, "observation_std": 0.1}
            fit_vae(vae, bundle, torch.device("cpu"), root / "vae", config)
            restored, _, _, device = restore_generative(root / "vae", "cpu")
            z = noise(2, 8, device, 2026)
            self.assertTrue(torch.allclose(vae.decode(z), restored.decode(z)))
            metrics = vae_epoch(restored, loader, device)
            self.assertEqual(metrics["samples"], 4)
            g, d = DCGenerator(8, 8), DCDiscriminator(8)
            config = dict(config, task="gan", lr=0.0002)
            fit_gan(g, d, bundle, device, root / "gan", config)
            g2, d2, _, _ = restore_generative(root / "gan", "cpu")
            g.eval()
            self.assertTrue(torch.allclose(g(z), g2(z)))
            result = gan_diagnostics(g2, d2, loader, device, 8)
            self.assertEqual(result["samples"], 4)
            self.assertIn("interpretation", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
