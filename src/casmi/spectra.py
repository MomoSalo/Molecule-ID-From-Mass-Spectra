"""Conservative cleaning and tolerance-based greedy cosine similarity."""

from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True)
class SearchConfig:
    precursor_ppm: float = 10.0
    precursor_da: float = 0.005
    fragment_da: float = 0.02
    intensity_floor: float = 0.001
    max_peaks: int = 128
    min_matched_peaks: int = 3

    def __post_init__(self):
        for name in ["precursor_ppm", "precursor_da", "fragment_da", "intensity_floor"]:
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.intensity_floor > 1 or self.max_peaks < 1 or self.min_matched_peaks < 1:
            raise ValueError("Invalid peak filtering or matching configuration")

    def to_dict(self) -> dict:
        return asdict(self)


def prepare(mzs, intensities, precursor_mz: float, config: SearchConfig):
    mz = np.asarray(mzs, dtype=np.float64)
    intensity = np.asarray(intensities, dtype=np.float64)
    if mz.ndim != 1 or intensity.ndim != 1 or mz.shape != intensity.shape:
        raise ValueError("Peak arrays must be aligned one-dimensional arrays")
    valid = np.isfinite(mz) & np.isfinite(intensity) & (mz > 0) & (intensity > 0)
    valid &= mz <= precursor_mz + 2.0
    mz, intensity = mz[valid], intensity[valid]
    if not len(mz):
        return mz, intensity
    intensity /= intensity.max()
    keep = intensity >= config.intensity_floor
    mz, intensity = mz[keep], intensity[keep]
    if len(mz) > config.max_peaks:
        # Stable sorting gives deterministic behavior for equal intensities.
        keep = np.argsort(-intensity, kind="stable")[:config.max_peaks]
        mz, intensity = mz[keep], intensity[keep]
    order = np.argsort(mz, kind="stable")
    mz, intensity = mz[order], np.sqrt(intensity[order])
    norm = np.linalg.norm(intensity)
    if norm:
        intensity /= norm
    return mz, intensity


def cosine(query_mz, query_i, ref_mz, ref_i, tolerance: float) -> tuple[float, int]:
    """Match each peak at most once, prioritizing the largest weight products.

    Inputs must be sorted by m/z, with nonnegative, L2-normalized weights.
    This is greedy cosine, not an optimal bipartite assignment.
    """
    pairs = []
    for q, mass in enumerate(query_mz):
        left = np.searchsorted(ref_mz, mass - tolerance, side="left")
        right = np.searchsorted(ref_mz, mass + tolerance, side="right")
        for r in range(left, right):
            pairs.append((float(query_i[q] * ref_i[r]), q, r))
    pairs.sort(key=lambda pair: (-pair[0], pair[1], pair[2]))
    used_q, used_r = set(), set()
    score = 0.0
    for product, q, r in pairs:
        if q not in used_q and r not in used_r:
            score += product
            used_q.add(q)
            used_r.add(r)
    return min(score, 1.0), len(used_q)
