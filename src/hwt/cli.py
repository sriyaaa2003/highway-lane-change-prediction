from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("PYTHONUTF8", "1")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="hwt")
    ap.add_argument("stage", choices=["ingest", "samples", "baselines", "bev", "evalcnn", "conformal", "mining", "energy", "figures", "report", "matched", "tune_hgb", "lolo", "sensitivity", "repro", "controls", "ensemble", "ensemble2", "portfolio", "label_audit"])
    ap.add_argument("--config", default="configs/base.yaml")
    a = ap.parse_args(argv)

    from .config import load_config
    cfg = load_config(a.config)
    if a.stage == "ingest":
        from . import ingest
        ingest.run(cfg)
    elif a.stage == "samples":
        from . import labels
        labels.run(cfg)
    elif a.stage == "bev":
        from . import bev
        bev.run(cfg)
    elif a.stage == "evalcnn":
        from . import evalcnn
        evalcnn.run(cfg)
    elif a.stage == "conformal":
        from . import conformal
        conformal.run(cfg)
    elif a.stage == "mining":
        from . import mining
        mining.run(cfg)
    elif a.stage == "energy":
        from . import energy
        if not (cfg.paths.results / 'energy_tracks.parquet').exists():
            energy.run(cfg)
        energy.analyze(cfg)
    elif a.stage == "figures":
        from . import viz
        viz.ladder(cfg)
        viz.coverage(cfg)
    elif a.stage == "report":
        from . import report
        report.run(cfg)
    elif a.stage == "matched":
        from . import matched
        matched.run(cfg)
    elif a.stage == "tune_hgb":
        from . import matched
        matched.tune_hgb(cfg, matched.load_bev(cfg))
    elif a.stage == "lolo":
        from . import lolo
        lolo.run(cfg)
    elif a.stage == "sensitivity":
        from . import sensitivity
        sensitivity.run(cfg)
    elif a.stage == "repro":
        from . import repro
        repro.run(cfg)
    elif a.stage == "controls":
        from . import controls
        controls.run(cfg)
    elif a.stage == "ensemble":
        from . import ensemble
        ensemble.run(cfg)
    elif a.stage == "ensemble2":
        from . import ensemble2
        ensemble2.run(cfg)
    elif a.stage == "portfolio":
        from . import portfolio_figs
        portfolio_figs.run(cfg)
    elif a.stage == "label_audit":
        from . import label_audit
        label_audit.run(cfg)
    elif a.stage == "baselines":
        from . import baselines
        baselines.run(cfg)


if __name__ == "__main__":
    main(sys.argv[1:])
