"""Modal wiring only. Run:  PYTHONUTF8=1 python -m modal run src/hwt/modal_app.py [--smoke]

Data must already be in the volume:  modal volume put hwt-data data/bev/highd.npz /highd.npz  (and exid.npz)
Outputs land in the volume under /out; fetch with  modal volume get hwt-data /out data/out
"""
import json

import modal

app = modal.App("hwt-bev")
vol = modal.Volume.from_name("hwt-data", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch", "numpy", "scikit-learn")
         .add_local_python_source("hwt"))


@app.function(image=image, gpu="T4", volumes={"/data": vol}, timeout=3600, memory=32768)
def train_one(spec: dict, cnn_cfg: dict) -> dict:
    import os

    import numpy as np

    from hwt.cnn_run import run_spec

    vol.reload()

    def load(ds):
        z = np.load(f"/data/{ds}.npz", allow_pickle=False)
        return {k: z[k] for k in z.files}

    def save(name, arrays):
        os.makedirs("/data/out", exist_ok=True)
        np.savez_compressed(f"/data/out/{name}.npz", **arrays)
        vol.commit()

    return run_spec(spec, load, save, cnn_cfg, "cuda")


@app.local_entrypoint()
def main(smoke: bool = False, tune: bool = False, controls: bool = False):
    from pathlib import Path

    from hwt.config import load_config

    cfg = load_config("configs/base.yaml")
    cc = cfg.cnn.model_dump()
    src, exid = [{"ds": "highd", "mask": "train"}], [{"ds": "exid", "mask": "train"}]
    if tune:      # validation-only: each grid entry trained on the highD-only and exiD-only training sets
        specs = [{"name": f"tune{i}_{n}", "train": t, "seed": 0, "cnn": over, "no_predict": True}
                 for i, over in enumerate(cc["tune_grid"]) for n, t in (("src", src), ("exid", exid))]
        res = list(train_one.map(specs, kwargs={"cnn_cfg": cc}))
        Path("results").mkdir(exist_ok=True)
        Path("results/cnn_tuning.json").write_text(json.dumps(
            [{**r, "override": cc["tune_grid"][int(r["name"][4:].split("_")[0])]} for r in res], indent=1))
        for r in res:
            print(json.dumps(r))
        return
    if controls:     # extra shuffled-label seeds: the null distribution of the CNN's score
        sel = Path("results/cnn_selected.json")
        cc.update(json.loads(sel.read_text()))
        specs = [{"name": f"{n}_s{s}", "train": t, "seed": s, "shuffle_labels": True}
                 for n, t in (("shuf", src), ("shufx", exid)) for s in (1, 2, 3, 4)]
        for r in train_one.map(specs, kwargs={"cnn_cfg": cc}):
            print(json.dumps(r))
        return
    sel = Path("results/cnn_selected.json")
    if sel.exists() and not smoke:
        cc.update(json.loads(sel.read_text()))
    ks = cfg.eval.k_fractions
    runs = [("src", src),
            *[(f"k{k}", src + [{"ds": "exid", "mask": f"kmask_{k}"}]) for k in ks],
            ("exid", exid)]
    seeds = [0] if smoke else cfg.cnn.seeds
    extra = {"max_train": 2000} if smoke else {}
    specs = [{"name": f"{n}_s{s}", "train": t, "seed": s, **extra} for n, t in runs for s in seeds]
    specs += [{"name": "rand_s0", "train": src, "seed": 0, "random_init": True, **extra},
              {"name": "shuf_s0", "train": src, "seed": 0, "shuffle_labels": True, **extra},
              {"name": "shufx_s0", "train": exid, "seed": 0, "shuffle_labels": True, **extra}]
    if smoke:
        cc["epochs"] = 1
    for r in train_one.map(specs, kwargs={"cnn_cfg": cc}):
        print(json.dumps(r))
