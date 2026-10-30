"""运行 CIFAR-10 图像分类实验。建议先 --smoke 检查环境。"""
import argparse
import json

import torch

from labkit.common import choose_device, seed_everything
from labkit.data import load_image_data
from labkit.engine import final_test, fit
from labkit.image_models import build_image_model, parameter_count
from labkit.plots import save_run_plots


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["smallcnn", "alexnet", "vgg11", "resnet18"], default="resnet18")
    p.add_argument("--variant", choices=["cifar", "torchvision"], default="cifar")
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--smoke", action="store_true", help="128/64/64 样本，仅检查流程，不用于性能比较")
    args = p.parse_args()
    if args.batch_size < 2 or args.lr <= 0:
        p.error("batch-size 至少为 2，lr 必须大于 0")
    seed_everything(args.seed)
    device = choose_device(args.device)
    model = build_image_model(args.model, args.variant, args.pretrained, args.freeze_backbone)
    bundle = load_image_data(args.data_dir, args.batch_size, args.seed, args.workers,
                             args.variant, not args.no_augment, not args.offline,
                             *([128, 64, 64] if args.smoke else [0, 0, 0]))
    config = dict(vars(args), task="image", parameters=parameter_count(model),
                  torch_version=str(torch.__version__))
    print(json.dumps({"device": str(device), "sizes": bundle.meta["sizes"],
                      "parameters": config["parameters"]}, indent=2), flush=True)
    model.to(device)
    optimizer = torch.optim.SGD([x for x in model.parameters() if x.requires_grad],
                                lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    epochs = 1 if args.smoke else args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history = fit(model, bundle, optimizer, device, args.run_dir, config, epochs,
                   amp=args.amp, grad_clip=5.0, freeze_backbone=args.freeze_backbone,
                   scheduler=scheduler)
    metrics = final_test(model, bundle, device, args.run_dir)
    save_run_plots(history, metrics, args.run_dir)


if __name__ == "__main__":
    main()
