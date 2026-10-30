"""运行 IMDB 英文影评情感分类实验。"""
import argparse
import json

import torch

from labkit.common import choose_device, seed_everything
from labkit.data import load_text_data
from labkit.engine import final_test, fit
from labkit.image_models import parameter_count
from labkit.plots import save_run_plots
from labkit.text_models import build_text_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["bilstm", "textcnn"], default="bilstm")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--max-vocab", type=int, default=20000)
    p.add_argument("--min-freq", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.batch_size < 1 or args.lr <= 0 or args.max_length < 5:
        p.error("batch-size、lr 必须为正，max-length 至少为 5")
    seed_everything(args.seed)
    device = choose_device(args.device)
    bundle = load_text_data(args.data_dir, args.batch_size, args.seed, args.workers,
                            args.max_vocab, args.min_freq, args.max_length,
                            not args.offline, *([128, 64, 64] if args.smoke else [0, 0, 0]))
    model = build_text_model(args.model, len(bundle.vocab))
    config = dict(vars(args), task="text", parameters=parameter_count(model),
                  torch_version=str(torch.__version__))
    print(json.dumps({"device": str(device), "sizes": bundle.meta["sizes"],
                      "vocab_size": len(bundle.vocab)}, indent=2), flush=True)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = fit(model, bundle, optimizer, device, args.run_dir, config,
                   1 if args.smoke else args.epochs, amp=args.amp, grad_clip=1.0)
    metrics = final_test(model, bundle, device, args.run_dir)
    save_run_plots(history, metrics, args.run_dir)


if __name__ == "__main__":
    main()
