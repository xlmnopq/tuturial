"""无需下载数据的功能测试。真实数据集 smoke 测试另见 tools/smoke_all.py。"""
import gc
import json
import math
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from labkit.common import (balanced_limit, fixed_split, load_checkpoint,
                           seed_everything, stratified_split)
from labkit.data import (DataBundle, Vocabulary, collate_reviews, image_transforms,
                         load_text_data, tokenize)
from labkit.engine import final_test, fit, run_epoch
from labkit.image_models import build_image_model
from labkit.inference import predict_text, restore_run
from labkit.text_models import BiLSTM, TextCNN, build_text_model


class DataTests(unittest.TestCase):
    def test_split_and_balanced_subset(self):
        labels = [0] * 50 + [1] * 50
        train, val = stratified_split(labels)
        self.assertEqual((len(train), len(val)), (90, 10))
        self.assertFalse(set(train) & set(val))
        self.assertEqual(sorted(train + val), list(range(100)))
        self.assertEqual((train, val), stratified_split(labels))
        limited = balanced_limit(train, labels, 8)
        self.assertEqual(sum(labels[i] for i in limited), 4)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "split.json"
            self.assertEqual(fixed_split(labels, path), fixed_split(labels, path))
            with self.assertRaises(ValueError):
                fixed_split(labels[::-1], path)

    def test_vocab_and_empty_sequences(self):
        vocab = Vocabulary.build(["great movie", "great acting"], min_freq=1)
        self.assertNotIn("only_in_test", vocab.stoi)
        self.assertEqual(vocab.encode(""), [1])
        tokens, lengths, labels = collate_reviews([(torch.tensor([1]), 0),
                                                   (torch.tensor([2, 3, 2]), 1)])
        self.assertEqual(tokens.shape, (2, 5))
        self.assertEqual(lengths.tolist(), [1, 3])
        self.assertEqual(tokens[0, 1:].sum().item(), 0)
        self.assertEqual(tokenize("Good<br />Movie!"), ["good", "movie", "!"])

    def test_vocabulary_uses_train_only(self):
        # 本地微型语料仅用于测试软件行为，不作为第三个课程数据集。
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "imdb"
            folder.mkdir()
            train = [{"id": str(i), "text": "common uniqueword" + chr(97 + i),
                      "label": i % 2} for i in range(20)]
            test = [{"id": str(i), "text": "testexclusive", "label": i % 2}
                    for i in range(4)]
            for name, rows in [("train", train), ("test", test)]:
                (folder / (name + ".jsonl")).write_text(
                    "\n".join(json.dumps(x) for x in rows), encoding="utf-8")
            bundle = load_text_data(tmp, min_freq=1, download=False)
            self.assertNotIn("testexclusive", bundle.vocab.stoi)
            for index in bundle.meta["val_indices"]:
                self.assertNotIn("uniqueword" + chr(97 + index), bundle.vocab.stoi)
            self.assertEqual(bundle.meta["sizes"], {"train": 18, "val": 2, "test": 4})

    def test_eval_transforms_are_deterministic(self):
        import numpy as np
        from PIL import Image
        image = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
        for variant, size in [("cifar", 32), ("torchvision", 224)]:
            _, transform, _, _ = image_transforms(variant)
            self.assertTrue(torch.equal(transform(image), transform(image)))
            self.assertEqual(tuple(transform(image).shape), (3, size, size))


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_all_image_models_forward_backward(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for name in ["smallcnn", "alexnet", "vgg11", "resnet18"]:
            with self.subTest(name=name):
                seed_everything(42)
                model = build_image_model(name).to(device)
                x = torch.randn(2, 3, 32, 32, device=device)
                y = torch.tensor([0, 1], device=device)
                logits = model(x)
                self.assertEqual(tuple(logits.shape), (2, 10))
                nn.CrossEntropyLoss()(logits, y).backward()
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                self.assertTrue(all(torch.isfinite(g).all().item() for g in grads))
                self.assertGreater(sum(g.abs().sum().item() for g in grads), 0)
                del model, logits, grads
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    def test_torchvision_native_models(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for name in ["alexnet", "vgg11", "resnet18"]:
            with self.subTest(name=name):
                model = build_image_model(name, variant="torchvision").to(device).eval()
                with torch.no_grad():
                    logits = model(torch.zeros(1, 3, 224, 224, device=device))
                self.assertEqual(tuple(logits.shape), (1, 10))
                del model, logits
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    def test_padding_invariance_and_text_backward(self):
        seed_everything(42)
        for model in [BiLSTM(30, embed_dim=8, hidden_dim=8, dropout=0),
                       TextCNN(30, embed_dim=8, num_channels=8, dropout=0)]:
            with self.subTest(model=type(model).__name__):
                model.eval()
                ids = torch.tensor([[2, 3, 4, 5, 6, 7]])
                lengths = torch.tensor([6])
                before = model(ids, lengths)
                padded = torch.nn.functional.pad(ids, (0, 11))
                after = model(padded, lengths)
                self.assertTrue(torch.allclose(before, after, atol=1e-6))
                nn.CrossEntropyLoss()(before, torch.tensor([1])).backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()
                                    if p.grad is not None))


class EngineTests(unittest.TestCase):
    def test_weighted_metrics_and_confusion(self):
        model = nn.Linear(3, 2)
        nn.init.zeros_(model.weight)
        nn.init.zeros_(model.bias)
        dataset = TensorDataset(torch.zeros(5, 3), torch.tensor([0, 0, 0, 1, 1]))
        metrics = run_epoch(model, DataLoader(dataset, batch_size=2), torch.device("cpu"))
        self.assertAlmostEqual(metrics["loss"], math.log(2), places=6)
        self.assertEqual(metrics["accuracy"], 0.6)
        self.assertEqual(metrics["confusion_matrix"], [[3, 0], [2, 0]])

    def test_fit_restore_and_no_test_during_fit(self):
        seed_everything(42)
        dataset = TensorDataset(torch.randn(8, 3), torch.tensor([0, 1] * 4))
        loader = DataLoader(dataset, batch_size=4)
        # test=None：如果 fit 错误地访问测试集，这项测试会失败。
        bundle = DataBundle(loader, loader, None, {"dataset": "imdb", "classes": ["neg", "pos"]})
        model = nn.Linear(3, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        with tempfile.TemporaryDirectory() as tmp:
            history = fit(model, bundle, optimizer, torch.device("cpu"), tmp,
                           {"task": "unit_test"}, epochs=2)
            self.assertEqual(len(history), 2)
            checkpoint = load_checkpoint(Path(tmp) / "best.pt")
            self.assertGreaterEqual(checkpoint["best_epoch"], 1)
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, checkpoint["model_state"][key]))
            with self.assertRaises(FileExistsError):
                fit(model, bundle, optimizer, torch.device("cpu"), tmp, {}, epochs=1)

    def test_real_text_checkpoint_contract(self):
        # 用微型语料实际训练、保存、恢复并预测，验证词表与权重的一致性。
        seed_everything(42)
        vocab = Vocabulary.build(["good movie", "bad movie"], min_freq=1)
        samples = [(torch.tensor(vocab.encode(t)), y)
                   for t, y in [("good movie", 1), ("bad movie", 0)] * 2]
        loader = DataLoader(samples, batch_size=2, collate_fn=collate_reviews)
        meta = {"dataset": "imdb", "classes": ["negative", "positive"]}
        bundle = DataBundle(loader, loader, loader, meta, vocab)
        model = build_text_model("textcnn", len(vocab))
        config = {"task": "text", "model": "textcnn", "max_length": 32}
        with tempfile.TemporaryDirectory() as tmp:
            fit(model, bundle, torch.optim.Adam(model.parameters()), torch.device("cpu"),
                tmp, config, epochs=1)
            final_test(model, bundle, torch.device("cpu"), tmp)
            restored, _, _, saved_vocab, _ = restore_run(tmp, "cpu")
            self.assertEqual(saved_vocab.tokens, vocab.tokens)
            result = predict_text(tmp, "good movie", "cpu")
            self.assertAlmostEqual(sum(result["probabilities"].values()), 1, places=6)
            self.assertIn(result["label"], meta["classes"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
