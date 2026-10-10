"""Recording-cluster bootstrap with exact weighted average precision.

Resampling recordings with replacement is equivalent to weighting every row by how often its recording was drawn,
so each score is sorted once and a replicate costs one cumulative sum. Ties are broken by row order; scores here are
continuous, and tests compare the result with scikit-learn.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def weighted_ap(y_sorted: np.ndarray, w_sorted: np.ndarray) -> float:
    """AP of rows already sorted by decreasing score, each counted w times."""
    pos = w_sorted * y_sorted
    denom = pos.sum()
    if denom <= 0:
        return float("nan")
    tp = np.cumsum(pos)
    seen = np.cumsum(w_sorted)
    return float((pos * tp / np.maximum(seen, 1e-12)).sum() / denom)


class Cluster:
    """One test set: labels, recording codes, and the sorted order of each score vector."""

    def __init__(self, y: np.ndarray, rec: np.ndarray):
        self.y = y.astype(np.float64)
        self.codes, self.uniq = pd.factorize(rec)
        self.n_rec = len(self.uniq)

    def order(self, score: np.ndarray) -> np.ndarray:
        return np.argsort(-score, kind="stable")

    def point(self, score: np.ndarray) -> float:
        o = self.order(score)
        return weighted_ap(self.y[o], np.ones(len(o)))

    def draws(self, B: int, seed: int) -> np.ndarray:
        """(B, n_rec) integer counts: how often each recording is drawn."""
        rng = np.random.default_rng(seed)
        picks = rng.integers(0, self.n_rec, (B, self.n_rec))
        return np.stack([np.bincount(p, minlength=self.n_rec) for p in picks])

    def replicates(self, score: np.ndarray, counts: np.ndarray) -> np.ndarray:
        o = self.order(score)
        y, c = self.y[o], self.codes[o]
        return np.array([weighted_ap(y, cnt[c].astype(np.float64)) for cnt in counts])


def ci(vals: np.ndarray) -> tuple[float, float]:
    return float(np.nanquantile(vals, 0.025)), float(np.nanquantile(vals, 0.975))


def paired_p(diff: np.ndarray) -> float:
    """Two-sided bootstrap p-value for 'difference is zero' (resolution 2/(B+1))."""
    d = diff[~np.isnan(diff)]
    return float(max(2 * min((d <= 0).mean(), (d >= 0).mean()), 2.0 / (len(d) + 1)))


def holm(p: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the original order."""
    order = np.argsort(p)
    adj = np.empty(len(p))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()
