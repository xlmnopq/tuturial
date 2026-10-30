import json
from pathlib import Path

from .common import choose_device, load_checkpoint
from .generative_models import ConvVAE, DCDiscriminator, DCGenerator


def restore_generative(run_dir, device="auto"):
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    device = choose_device(device)
    if config["task"] == "vae":
        checkpoint = load_checkpoint(run_dir / "best.pt")
        model = ConvVAE(config["latent_dim"], config["channels"])
        model.load_state_dict(checkpoint["model_state"])
        return model.to(device).eval(), None, config, device
    if config["task"] == "gan":
        checkpoint = load_checkpoint(run_dir / "last.pt")
        generator = DCGenerator(config["latent_dim"], config["channels"])
        discriminator = DCDiscriminator(config["channels"])
        generator.load_state_dict(checkpoint["generator_state"])
        discriminator.load_state_dict(checkpoint["discriminator_state"])
        return generator.to(device).eval(), discriminator.to(device).eval(), config, device
    raise ValueError("运行目录需要来自 train_vae.py 或 train_gan.py")
