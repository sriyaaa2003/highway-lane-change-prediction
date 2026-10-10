"""BEV CNN: three horizon heads (1, 2, 3 s) over a raster + scalar features. Torch only; runs on CPU or GPU.

One network with three sigmoid heads is used instead of one network per horizon (blueprint ADR-6 amended) to save
GPU time; the heads share the representation, the labels per horizon stay independent.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn

HORIZONS = (1.0, 2.0, 3.0)


class BevNet(nn.Module):
    def __init__(self, n_scalar: int, widths=(32, 64, 96), embed_dim: int = 64, n_heads: int = 3):
        super().__init__()
        layers, c = [], 3
        for w in widths:
            layers += [nn.Conv2d(c, w, 3, padding=1), nn.BatchNorm2d(w), nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            c = w
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((6, 4))
        self.fc_img = nn.Sequential(nn.Flatten(), nn.Linear(c * 24, 128), nn.ReLU(inplace=True), nn.Dropout(0.2))
        self.fc_sc = nn.Sequential(nn.Linear(n_scalar, 64), nn.ReLU(inplace=True))
        self.embed = nn.Sequential(nn.Linear(128 + 64, embed_dim), nn.ReLU(inplace=True))
        self.head = nn.Linear(embed_dim, n_heads)

    def forward(self, x, s, return_embedding: bool = False):
        h = self.fc_img(self.pool(self.conv(x)))
        e = self.embed(torch.cat([h, self.fc_sc(s)], dim=1))
        return (self.head(e), e) if return_embedding else self.head(e)


def horizon_labels(ev: np.ndarray, ttl: np.ndarray) -> np.ndarray:
    t = np.nan_to_num(ttl, nan=np.inf)
    return np.stack([((t <= h + 1e-9) & (ev > 0)) for h in HORIZONS], axis=1).astype(np.float32)


def _batches(n: int, bs: int, shuffle: bool, gen: torch.Generator | None = None):
    order = torch.randperm(n, generator=gen) if shuffle else torch.arange(n)
    for i in range(0, n, bs):
        yield order[i:i + bs]


@torch.no_grad()
def predict(model: BevNet, X: np.ndarray, S: np.ndarray, device: str, bs: int = 2048, embeddings: bool = False):
    model.eval()
    probs, embs = [], []
    for i in range(0, len(X), bs):
        x = torch.from_numpy(X[i:i + bs]).to(device).float().div_(255.0)
        s = torch.from_numpy(S[i:i + bs]).to(device)
        lg, e = model(x, s, return_embedding=True)
        probs.append(torch.sigmoid(lg).cpu().numpy())
        if embeddings:
            embs.append(e.cpu().numpy())
    return (np.concatenate(probs), np.concatenate(embs) if embeddings else None)


def val_score(model, X, S, Y, device) -> float:
    P, _ = predict(model, X, S, device)
    ap = [average_precision_score(Y[:, k], P[:, k]) for k in range(Y.shape[1]) if Y[:, k].any()]
    return float(np.mean(ap)) if ap else 0.0


def train(Xtr, Str, Ytr, Xva, Sva, Yva, cnn_cfg: dict, seed: int, device: str, log=print):
    """Train with early stopping on mean validation PR-AUC over the three horizons."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = BevNet(Str.shape[1], tuple(cnn_cfg["widths"]), cnn_cfg["embed_dim"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cnn_cfg["lr"], weight_decay=cnn_cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cnn_cfg["lr"], total_steps=cnn_cfg["epochs"] *
                                                ((len(Xtr) + cnn_cfg["batch_size"] - 1) // cnn_cfg["batch_size"]))
    pos = Ytr.mean(0).clip(1e-4)
    pw = torch.tensor(np.minimum((1 - pos) / pos, cnn_cfg["pos_weight_cap"]), dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    Xg = torch.from_numpy(Xtr).to(device)             # uint8 on device; scaled per batch
    Sg, Yg = torch.from_numpy(Str).to(device), torch.from_numpy(Ytr).to(device)
    best, best_state, bad, t0 = -1.0, None, 0, time.time()
    for ep in range(cnn_cfg["epochs"]):
        model.train()
        tot = 0.0
        for b in _batches(len(Xtr), cnn_cfg["batch_size"], True, gen):
            b = b.to(device)
            loss = loss_fn(model(Xg[b].float().div_(255.0), Sg[b]), Yg[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss.detach()) * len(b)
        v = val_score(model, Xva, Sva, Yva, device)
        log(f"epoch {ep + 1} loss {tot / len(Xtr):.4f} val PR-AUC {v:.4f} ({time.time() - t0:.0f}s)")
        if v > best:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
        if bad >= cnn_cfg["patience"] or (time.time() - t0) / 60 > cnn_cfg["max_minutes"]:
            break
    model.load_state_dict(best_state)
    return model, best
