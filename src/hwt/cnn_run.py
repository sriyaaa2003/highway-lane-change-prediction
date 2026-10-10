"""Run one CNN training spec against the BEV npz files. Shared by Modal and local smoke tests."""
from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from . import cnn

EVAL_SPLITS = ("cal", "test", "holdout_loc")


def _gather(d: dict, parts: list[dict]):
    """parts: [{'ds': 'highd', 'mask': 'train' | 'kmask_0.05'}...]. Returns X, S, Y concatenated."""
    Xs, Ss, Ys = [], [], []
    for p in parts:
        data = d[p["ds"]]
        m = data[p["mask"]] if p["mask"].startswith("kmask") else data["split"] == p["mask"]
        Xs.append(data["X"][m])
        Ss.append(data["S"][m])
        Ys.append(cnn.horizon_labels(data["ev"][m], data["ttl"][m]))
    return np.concatenate(Xs), np.concatenate(Ss), np.concatenate(Ys)


def run_spec(spec: dict, load: Callable[[str], dict], save: Callable[[str, dict], None], cnn_cfg: dict,
             device: str, log=print) -> dict:
    d = {ds: load(ds) for ds in ("highd", "exid")}
    seed = spec["seed"]
    cnn_cfg = {**cnn_cfg, **spec.get("cnn", {})}          # per-spec overrides (tuning grid, selected config)
    cap = spec.get("max_train")              # smoke tests only
    t0 = time.time()
    train_ds = {p["ds"] for p in spec["train"]}
    Xtr, Str, Ytr = _gather(d, spec["train"])
    Xva, Sva, Yva = _gather(d, [{"ds": ds, "mask": "val"} for ds in sorted(train_ds)])
    if cap:
        rng = np.random.default_rng(0)
        keep = rng.choice(len(Xtr), min(cap, len(Xtr)), replace=False)
        Xtr, Str, Ytr = Xtr[keep], Str[keep], Ytr[keep]
        Xva, Sva, Yva = Xva[:cap], Sva[:cap], Yva[:cap]
    if spec.get("shuffle_labels"):                          # control: labels carry no information
        rng2 = np.random.default_rng(1000 + seed)
        Ytr, Yva = Ytr[rng2.permutation(len(Ytr))], Yva[rng2.permutation(len(Yva))]
    mu, sd = Str.mean(0), Str.std(0) + 1e-6
    norm = lambda s: ((s - mu) / sd).astype(np.float32)
    log(f"[{spec['name']}] train {len(Xtr)} (pos/h {Ytr.mean(0).round(4).tolist()}) val {len(Xva)}")

    if spec.get("random_init"):
        import torch
        torch.manual_seed(seed)
        model, best = cnn.BevNet(Str.shape[1], tuple(cnn_cfg["widths"]), cnn_cfg["embed_dim"]).to(device), float("nan")
    else:
        model, best = cnn.train(Xtr, norm(Str), Ytr, Xva, norm(Sva), Yva, cnn_cfg, seed, device, log)

    if spec.get("no_predict"):                              # tuning runs only need the validation score
        return {"name": spec["name"], "val_pr_auc": best, "n_train": len(Xtr), "wall_s": time.time() - t0}
    out = {}
    for ds in ("highd", "exid"):
        data = d[ds]
        idx = np.flatnonzero(np.isin(data["split"], EVAL_SPLITS))
        if cap:
            idx = idx[:cap]
        P, E = cnn.predict(model, data["X"][idx], norm(data["S"][idx]), device, embeddings=True)
        out[f"{ds}_idx"], out[f"{ds}_P"], out[f"{ds}_E"] = idx, P.astype(np.float16), E.astype(np.float16)
    save(spec["name"], out)
    return {"name": spec["name"], "val_pr_auc": best, "n_train": len(Xtr), "wall_s": time.time() - t0}
