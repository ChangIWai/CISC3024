"""
CISC3024 AI Assignment #2
Bayesian image classification on CIFAR-10:
  1. MAP CNN              (standard training, softmax output)
  2. MC Dropout           (Gal & Ghahramani, 2016)
  3. Last-layer Laplace   (Daxberger et al., NeurIPS 2021, "Laplace Redux")

Metrics: accuracy, NLL, ECE, Brier score, and OOD detection AUROC (CIFAR-10 vs SVHN).

Usage:
    pip install torch torchvision laplace-torch scikit-learn matplotlib numpy
    python bayes_cifar10.py --epochs 30 --seeds 3
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from laplace import Laplace
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def get_loaders(root, batch_size, workers):
    norm = T.Normalize(MEAN, STD)
    train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), norm])
    test_tf = T.Compose([T.ToTensor(), norm])

    train_set = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=train_tf)
    fit_set = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=test_tf)
    test_set = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=test_tf)
    ood_set = torchvision.datasets.SVHN(root, split="test", download=True, transform=test_tf)

    kw = dict(num_workers=workers, pin_memory=torch.cuda.is_available())
    return dict(
        train=DataLoader(train_set, batch_size, shuffle=True, drop_last=True, **kw),
        fit=DataLoader(fit_set, batch_size, shuffle=False, **kw),  # un-augmented, for Laplace
        test=DataLoader(test_set, 256, shuffle=False, **kw),
        ood=DataLoader(ood_set, 256, shuffle=False, **kw),
    )


# ----------------------------------------------------------------------------
# Model: a small VGG-style CNN. Final layer is a single nn.Linear so that
# last-layer Laplace can be applied directly.
# ----------------------------------------------------------------------------
def conv_block(cin, cout, p):
    layers = [
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    ]
    if p > 0:
        layers.append(nn.Dropout(p))
    return layers


def make_cnn(p_drop=0.0):
    return nn.Sequential(
        *conv_block(3, 64, p_drop),
        *conv_block(64, 128, p_drop),
        *conv_block(128, 256, p_drop),
        nn.Flatten(),
        nn.Linear(256 * 4 * 4, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p_drop) if p_drop > 0 else nn.Identity(),
        nn.Linear(256, 10),
    )


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def train(model, loader, epochs, lr, wd, device):
    model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(loader))
    for ep in range(epochs):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            opt.step()
            sched.step()
            total += y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            loss_sum += loss.item() * y.size(0)
        print(f"  epoch {ep + 1:3d}/{epochs}  loss {loss_sum / total:.4f}  train acc {correct / total:.4f}")
    return model


# ----------------------------------------------------------------------------
# Predictive distributions
# ----------------------------------------------------------------------------
@torch.no_grad()
def predict_map(model, loader, device):
    model.eval()
    return torch.cat([F.softmax(model(x.to(device)), -1).cpu() for x, _ in loader])


@torch.no_grad()
def predict_mc_dropout(model, loader, device, T_samples=30):
    model.eval()
    for m in model.modules():  # keep only dropout stochastic (BatchNorm stays in eval mode)
        if isinstance(m, nn.Dropout):
            m.train()
    probs = []
    for x, _ in loader:
        x = x.to(device)
        p = torch.stack([F.softmax(model(x), -1) for _ in range(T_samples)]).mean(0)
        probs.append(p.cpu())
    return torch.cat(probs)


@torch.no_grad()
def predict_laplace(la, loader, device):
    probs = []
    for x, _ in loader:
        probs.append(la(x.to(device), link_approx="probit").cpu())  # closed-form predictive
    return torch.cat(probs)


def fit_laplace(model, fit_loader):
    model.eval()
    la = Laplace(model, "classification", subset_of_weights="last_layer", hessian_structure="kron")
    la.fit(fit_loader)
    la.optimize_prior_precision(method="marglik")  # tune lambda by maximising the evidence
    return la


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def labels_of(loader):
    return torch.cat([y for _, y in loader])


def ece_score(probs, y, n_bins=15):
    conf, pred = probs.max(1)
    acc = (pred == y).float()
    edges = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.float().mean() * (acc[m].mean() - conf[m].mean()).abs()
    return float(ece)


def evaluate(probs, y, ood_probs):
    nll = F.nll_loss(torch.log(probs.clamp_min(1e-12)), y).item()
    brier = ((probs - F.one_hot(y, 10).float()) ** 2).sum(1).mean().item()
    scores = torch.cat([probs.max(1).values, ood_probs.max(1).values]).numpy()
    is_id = np.concatenate([np.ones(len(probs)), np.zeros(len(ood_probs))])
    return dict(
        acc=(probs.argmax(1) == y).float().mean().item() * 100,
        nll=nll,
        ece=ece_score(probs, y) * 100,
        brier=brier,
        auroc=roc_auc_score(is_id, scores) * 100,
    )


def reliability_plot(results, y, path, n_bins=15):
    fig, axes = plt.subplots(1, len(results), figsize=(4 * len(results), 4), sharey=True)
    for ax, (name, probs) in zip(np.atleast_1d(axes), results.items()):
        conf, pred = probs.max(1)
        acc = (pred == y).float()
        edges = torch.linspace(0, 1, n_bins + 1)
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.any():
                xs.append(conf[m].mean().item())
                ys.append(acc[m].mean().item())
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.plot(xs, ys, "o-")
        ax.set_title(f"{name}\nECE {ece_score(probs, y) * 100:.2f}%")
        ax.set_xlabel("confidence")
    np.atleast_1d(axes)[0].set_ylabel("accuracy")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data")
    ap.add_argument("--out", default="./results")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--mc-samples", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"device: {device}  args: {vars(args)}")

    loaders = get_loaders(args.data, args.batch_size, args.workers)
    y_test = labels_of(loaders["test"])
    all_runs = {"MAP CNN": [], f"MC Dropout (p={args.dropout})": [], "Last-layer Laplace": []}
    names = list(all_runs)

    for seed in range(args.seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        t0 = time.time()

        print(f"\n=== seed {seed}: MAP CNN ===")
        map_model = train(make_cnn(0.0), loaders["train"], args.epochs, args.lr, args.wd, device)
        p_map = predict_map(map_model, loaders["test"], device)
        o_map = predict_map(map_model, loaders["ood"], device)

        print(f"\n=== seed {seed}: last-layer Laplace (post-hoc on the MAP CNN) ===")
        t1 = time.time()
        la = fit_laplace(map_model, loaders["fit"])
        print(f"  Laplace fit time: {time.time() - t1:.1f}s  prior precision: {la.prior_precision}")
        p_la = predict_laplace(la, loaders["test"], device)
        o_la = predict_laplace(la, loaders["ood"], device)

        print(f"\n=== seed {seed}: MC Dropout ===")
        mc_model = train(make_cnn(args.dropout), loaders["train"], args.epochs, args.lr, args.wd, device)
        p_mc = predict_mc_dropout(mc_model, loaders["test"], device, args.mc_samples)
        o_mc = predict_mc_dropout(mc_model, loaders["ood"], device, args.mc_samples)

        for name, (p, o) in zip(names, [(p_map, o_map), (p_mc, o_mc), (p_la, o_la)]):
            r = evaluate(p, y_test, o)
            all_runs[name].append(r)
            print(f"seed {seed} | {name:28s} | " + "  ".join(f"{k} {v:.3f}" for k, v in r.items()))

        if seed == 0:
            reliability_plot(dict(zip(names, [p_map, p_mc, p_la])), y_test, out / "reliability.png")
        print(f"seed {seed} finished in {(time.time() - t0) / 60:.1f} min")

    # Summary table (mean +- std over seeds)
    keys = ["acc", "nll", "ece", "brier", "auroc"]
    header = "| Method | Acc (%) | NLL | ECE (%) | Brier | OOD AUROC (%) |\n| --- | --- | --- | --- | --- | --- |"
    rows = []
    for name, runs in all_runs.items():
        cells = []
        for k in keys:
            v = np.array([r[k] for r in runs])
            cells.append(f"{v.mean():.2f} ± {v.std():.2f}" if k != "nll" and k != "brier" else f"{v.mean():.3f} ± {v.std():.3f}")
        rows.append(f"| {name} | " + " | ".join(cells) + " |")
    table = header + "\n" + "\n".join(rows)
    print("\n" + table)
    (out / "summary.md").write_text(table + "\n")
    (out / "raw_results.json").write_text(json.dumps(all_runs, indent=2))


if __name__ == "__main__":
    main()
