"""Markdown tables generated from results/*.csv and results/energy_summary.json (no numbers typed by hand)."""
from __future__ import annotations

import json

import pandas as pd

from .config import Config


def _f(r, k="pr_auc"):
    return f"{r[k]:.2f} [{r[k + '_lo'] if k + '_lo' in r else r['ci_lo']:.2f}, {r[k + '_hi'] if k + '_hi' in r else r['ci_hi']:.2f}]"


def _cell(row, m):
    return f"{row[f'ap_{m}']:.2f} [{row[f'ap_{m}_lo']:.2f}, {row[f'ap_{m}_hi']:.2f}]"


def ladder_table(cfg: Config) -> str:
    d = pd.read_csv(cfg.paths.results / "matched.csv")
    d = d[d.H == cfg.eval.primary_horizon_s]
    rows = [{"rung": r.rung, "test": r.test, "subset": r.subset, "positives": r.positives, "prevalence": round(r.prevalence, 3),
             "rule": _cell(r, "rule"), "boosting": _cell(r, "hgb"), "CNN": _cell(r, "cnn")} for r in d.itertuples()
            for r in [d.loc[r.Index]]]
    return pd.DataFrame(rows).to_markdown(index=False)


def paired_table(cfg: Config) -> str:
    d = pd.read_csv(cfg.paths.results / "matched.csv")
    d = d[d.H == cfg.eval.primary_horizon_s]
    rows = [{"rung": r.rung, "test": r.test, "subset": r.subset, "positives": r.positives,
             "CNN - boosting": f"{r.d_cnn_hgb:+.3f} [{r.d_cnn_hgb_lo:+.3f}, {r.d_cnn_hgb_hi:+.3f}]", "p (Holm)": f"{r.p_cnn_hgb_holm:.3f}",
             "boosting - rule": f"{r.d_hgb_rule:+.3f} [{r.d_hgb_rule_lo:+.3f}, {r.d_hgb_rule_hi:+.3f}]", "p (Holm) ": f"{r.p_hgb_rule_holm:.3f}"}
            for r in d.itertuples()]
    return pd.DataFrame(rows).to_markdown(index=False)


def controls_table(cfg: Config) -> str:
    c = pd.read_csv(cfg.paths.results / "controls.csv")
    m = pd.read_csv(cfg.paths.results / "matched.csv")
    m = m[(m.H == cfg.eval.primary_horizon_s) & m.ap_hgb_shuf.notna()].set_index(["test", "subset"]).sort_index()
    rows = []
    for r in c.itertuples():
        try:
            hs = _cell(m.loc[(r.test, r.subset)].iloc[0], "hgb_shuf")
        except KeyError:
            hs = "n/a"
        rows.append({"trained on": r.trained_on, "test": r.test, "subset": r.subset, "prevalence": round(r.prevalence, 3),
                     "boosting, shuffled": hs, "CNN shuffled: mean": round(r.ap_mean, 3),
                     "CNN shuffled: per seed": str(r.per_seed)})
    return pd.DataFrame(rows).to_markdown(index=False)


def lolo_table(cfg: Config) -> str:
    d = pd.read_csv(cfg.paths.results / "lolo.csv")
    rows = [{"dataset": r.dataset, "held-out location": r["held_out_location"], "subset": r.subset, "recordings": r.recordings,
             "positives": r.positives, "rule": _cell(r, "rule"), "boosting": _cell(r, "hgb")} for _, r in d.iterrows()]
    return pd.DataFrame(rows).to_markdown(index=False)


def sensitivity_table(cfg: Config) -> str:
    d = pd.read_csv(cfg.paths.results / "sensitivity.csv")
    rows = [{"variant": r["variant"], "rung": r["rung"], "subset": r.subset, "positives": r.positives,
             "rule": _cell(r, "rule"), "boosting": _cell(r, "hgb")} for _, r in d.iterrows()]
    return pd.DataFrame(rows).to_markdown(index=False)


def tuning_tables(cfg: Config) -> str:
    h = pd.read_csv(cfg.paths.results / "hgb_tuning.csv").sort_values("val_ap_mean", ascending=False).round(4)
    c = json.loads((cfg.paths.results / "cnn_tuning.json").read_text())
    ct = pd.DataFrame([{"override": json.dumps(x["override"]), "domain": x["name"].split("_")[1], "val_pr_auc": round(x["val_pr_auc"], 4)}
                       for x in c]).pivot_table(index="override", columns="domain", values="val_pr_auc")
    ct["mean"] = ct.mean(axis=1)
    rep = json.loads((cfg.paths.results / "repro.json").read_text())
    sep = chr(10) * 2
    return sep.join(["Boosting (validation AP at 2 s):" + sep + h.to_markdown(index=False),
                     "CNN (validation PR-AUC averaged over horizons):" + sep
                     + ct.sort_values("mean", ascending=False).round(4).to_markdown(),
                     "Reproducibility check: " + json.dumps(rep)])


def ensemble_table(cfg: Config) -> str:
    d = pd.read_csv(cfg.paths.results / "ensemble.csv")
    m = pd.read_csv(cfg.paths.results / "ensemble_models.csv")
    rows = []
    for _, r in d.iterrows():
        row = {"config": r["config"], "test": r["test"], "subset": r["subset"], "positives": r["positives"],
               "rule": f"{r['ap_rule']:.2f}", "boosting": _cell(r, "gen"), "CNN (3-seed mean score)": f"{r['ap_cnn']:.2f}"}
        for k, name in (("hier", "hierarchy"), ("stack3", "stack: rule+boost+CNN"), ("stack4", "stack + specialist")):
            row[name] = f"{r['ap_' + k]:.2f} ({r['d_' + k + '_gen']:+.3f}, p={r['p_' + k + '_gen_holm']:.3f})"
        rows.append(row)
    sep = chr(10) * 2
    return pd.DataFrame(rows).to_markdown(index=False) + sep + "Specialists:" + sep + m.to_markdown(index=False)


def ensemble2_tables(cfg: Config) -> str:
    sel = pd.read_csv(cfg.paths.results / "ensemble2_selection.csv")
    cv = [c for c in sel.columns if c.startswith("cv_")]
    t = sel.set_index("config")[cv].T
    t.index = [i[3:] for i in t.index]
    d = pd.read_csv(cfg.paths.results / "ensemble2.csv")
    rows = [{"config": r["config"], "test": r["test"], "subset": r["subset"], "positives": r["positives"], "selected": r["selected"],
             "boosting AP": f"{r['ap_gen']:.3f}", "selected - boosting": f"{r['d_sel_gen']:+.3f} [{r['d_lo']:+.3f}, {r['d_hi']:+.3f}]",
             "p (Holm)": f"{r['p_holm']:.3f}"} for _, r in d.iterrows()]
    vs = [c[3:] for c in d.columns if c.startswith("ap_") and c not in ("ap_gen", "ap_et", "ap_lr", "ap_res", "ap_rule", "ap_cnn")]
    desc = []
    for v in vs:
        row = {"variant": v}
        for sub in ("all", "pre_drift"):
            x = d[d.subset == sub]
            dl = x["ap_" + v] - x["ap_gen"]
            row[f"{sub}: mean diff to boosting"] = f"{dl.mean():+.3f}"
            row[f"{sub}: cells better"] = f"{int((dl > 0).sum())}/{len(x)}"
        desc.append(row)
    sep = chr(10) * 2
    return sep.join(["Calibration cross-validation score per variant (higher is better; selection uses only this):",
                     t.round(3).to_markdown(),
                     "Selected variant tested against plain boosting (paired bootstrap, Holm over 20 cells):",
                     pd.DataFrame(rows).to_markdown(index=False),
                     "All variants on test, descriptive only (no significance claims):",
                     pd.DataFrame(desc).to_markdown(index=False)])


def conformal_table(cfg: Config) -> str:
    d = pd.read_csv(cfg.paths.results / "conformal.csv")
    rows = []
    for _, r in d.iterrows():
        rows.append({"scenario": r.scenario, "test": r.test, "subset": r.subset, "test positives": r.test_pos,
                     "coverage (all)": f"{r['cov']:.2f} [{r['cov_lo']:.2f}, {r['cov_hi']:.2f}]",
                     "coverage (lane changes)": f"{r.cov_pos:.2f} [{r.cov_pos_lo:.2f}, {r.cov_pos_hi:.2f}]",
                     "mean set size": round(r["size"], 2), "empty sets": f"{r.empty:.0%}"})
    return pd.DataFrame(rows).to_markdown(index=False)


def mining_table(cfg: Config) -> str:
    d = pd.read_csv(cfg.paths.results / "mining.csv")
    d["P@k"] = d.apply(lambda r: f"{r.precision_at_k:.2f} [{r.lo:.2f}, {r.hi:.2f}]", axis=1)
    t = d.pivot_table(index=["scenario", "tag", "prevalence", "pool_pos"], columns="embedding", values="P@k", aggfunc="first")
    return t.reset_index().round(3).to_markdown(index=False)


def energy_tables(cfg: Config) -> str:
    s = json.loads((cfg.paths.results / "energy_summary.json").read_text())
    sens = pd.read_csv(cfg.paths.results / "energy_sensitivity.csv")
    sens = sens.reindex(sens.change_pct.abs().sort_values(ascending=False).index).head(8)
    cl = pd.read_csv(cfg.paths.results / "energy_clusters.csv")
    return "\n\n".join([
        f"Tracks: {s['n_tracks']}. Modelled mean {s['mean_wh_km_base']} Wh/km. By dataset: {s['by_dataset']}.",
        "Clusters (per k): " + json.dumps({k: {a: round(b, 3) for a, b in v.items()} for k, v in s["clusters"]["per_k"].items()}),
        f"Chosen k = {s['clusters']['chosen_k']}, stability criterion met: {s['clusters']['stable']}.",
        cl.to_markdown(index=False),
        "Style beyond speed: " + json.dumps(s["style_beyond_speed"]),
        "Acceleration source: " + json.dumps(s["accel_source"]),
        "Largest sensitivities (+-20% on one parameter):\n\n" + sens.to_markdown(index=False)])


def run(cfg: Config) -> None:
    parts = {"Shift ladder on matched samples, 2 s horizon, PR-AUC [95% CI]": ladder_table(cfg),
             "Paired differences at 2 s (Holm-adjusted over all cells)": paired_table(cfg),
             "Shuffled-label controls (2 s)": controls_table(cfg),
             "Leave-one-location-out (boosting, 2 s)": lolo_table(cfg),
             "Sensitivity to label and pre-drift definitions (boosting, 2 s)": sensitivity_table(cfg),
             "Ensembles and hierarchy at 2 s (difference to boosting, Holm p over 60 tests)": ensemble_table(cfg),
             "13 ensemble and hierarchy variants, selected on the calibration split": ensemble2_tables(cfg),
             "Conformal prediction (alpha 0.1, 2 s horizon)": conformal_table(cfg),
             "Scenario mining, precision@k [query-draw 95% interval]": mining_table(cfg),
             "Energy (modelled)": energy_tables(cfg),
             "Tuning and reproducibility": tuning_tables(cfg)}
    sep = chr(10) * 2
    text = sep.join(f"## {k}" + sep + v for k, v in parts.items())
    (cfg.paths.results / "tables.md").write_text(text, encoding="utf-8")
