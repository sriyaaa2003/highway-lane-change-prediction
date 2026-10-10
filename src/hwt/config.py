"""Typed, validated configuration. Unknown keys are errors."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Paths(_Strict):
    raw: Path
    canonical: Path
    interim: Path
    results: Path


class DataCfg(_Strict):
    frame_rate: float
    history_s: float
    horizons_s: list[float]
    sample_stride_frames: int
    exclude_after_event_s: float
    event_window_s: float
    min_lateral_disp_m: float
    axis_sample_s: float
    max_samples_per_recording: int
    speed_sanity_max_ms: float
    exid_heading_unit: Literal["auto", "rad", "deg"]


class SplitFrac(_Strict):
    train: float
    val: float
    cal: float
    test: float


class SplitsCfg(_Strict):
    seed: int
    frac: SplitFrac
    highd_holdout_locations: list[int]
    exid_holdout_locations: list[int]


class FeaturesCfg(_Strict):
    hist_steps_s: list[float]
    ref_from_s: float
    ref_to_s: float
    lat_vel_window_s: float
    lane_width_m: float
    neighbor_range_m: float


class HGB(_Strict):
    max_iter: int
    learning_rate: float
    max_leaf_nodes: int
    l2_regularization: float
    class_weight: str | None


class ModelCfg(_Strict):
    hgb: HGB
    hgb_grid: dict[str, list]
    seeds: list[int]


class EvalCfg(_Strict):
    bootstrap_resamples: int
    paired_resamples: int
    min_events_per_class: int
    primary_horizon_s: float
    k_fractions: list[float]
    bootstrap_seed: int
    pre_drift_max_disp_1s_m: float


class BevCfg(_Strict):
    ahead_m: float
    behind_m: float
    half_width_m: float
    cell_long_m: float
    cell_lat_m: float
    speed_clip_ms: float
    per_recording_cap: int
    fill_gap_m: float
    fill_thw_s: float
    fill_ttc_s: float


class CnnCfg(_Strict):
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    widths: list[int]
    embed_dim: int
    seeds: list[int]
    patience: int
    pos_weight_cap: float
    max_minutes: float
    gpu: str
    tune_grid: list[dict]


class TagsCfg(_Strict):
    brake_decel_ms2: float
    brake_window_s: float
    ttc_crit_s: float
    ttc_window_s: float
    lane_change_horizon_s: float


class MiningCfg(_Strict):
    n_query: int
    k: int
    n_draws: int


class Config(_Strict):
    paths: Paths
    data: DataCfg
    splits: SplitsCfg
    features: FeaturesCfg
    model: ModelCfg
    eval: EvalCfg
    bev: BevCfg
    cnn: CnnCfg
    tags: TagsCfg
    mining: MiningCfg

    @property
    def hmax_frames(self) -> int:
        return round(max(self.data.horizons_s) * self.data.frame_rate)


def load_config(path: str | Path, root: Path | None = None) -> Config:
    root = root or Path(path).resolve().parent.parent
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for k, v in raw["paths"].items():
        raw["paths"][k] = str(root / v)
    return Config(**raw)
