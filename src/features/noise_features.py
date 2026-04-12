from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, skew


def extract_noise(image_channel: np.ndarray, denoised_channel: np.ndarray) -> np.ndarray:
    return image_channel.astype(np.float64, copy=False) - denoised_channel.astype(np.float64, copy=False)


def _safe_corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return 0.0
    value = np.corrcoef(a, b)[0, 1]
    if np.isnan(value) or np.isinf(value):
        return 0.0
    return float(value)


def _safe_skew(x: np.ndarray) -> float:
    value = skew(x, bias=False, nan_policy="omit")
    return float(np.nan_to_num(value))


def _safe_kurtosis(x: np.ndarray) -> float:
    value = kurtosis(x, fisher=True, bias=False, nan_policy="omit")
    return float(np.nan_to_num(value))


def extract_15d_features_from_channel_noise(noise: np.ndarray) -> np.ndarray:
    if noise.ndim != 2:
        raise ValueError("Expected 2D noise map for one channel")

    m, n = noise.shape
    if m < 2 or n < 2:
        raise ValueError("Need at least 2x2 noise map for correlation-based features")

    row_avg = np.mean(noise, axis=0)
    col_avg = np.mean(noise, axis=1)

    rho_row = np.array([_safe_corrcoef(row_avg, noise[row_idx, :]) for row_idx in range(m)], dtype=np.float64)
    rho_col = np.array([_safe_corrcoef(col_avg, noise[:, col_idx]) for col_idx in range(n)], dtype=np.float64)

    f1 = float(np.mean(rho_row))
    f2 = float(np.std(rho_row))
    f3 = _safe_skew(rho_row)
    f4 = _safe_kurtosis(rho_row)

    f5 = float(np.mean(rho_col))
    f6 = float(np.std(rho_col))
    f7 = _safe_skew(rho_col)
    f8 = _safe_kurtosis(rho_col)

    f9 = float(np.std(row_avg))
    f10 = _safe_skew(row_avg)
    f11 = _safe_kurtosis(row_avg)

    f12 = float(np.std(col_avg))
    f13 = _safe_skew(col_avg)
    f14 = _safe_kurtosis(col_avg)

    denom = f1 if abs(f1) > 1e-12 else 1e-12
    f15 = float((1.0 - (f5 / denom)) * 100.0)

    return np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15], dtype=np.float64)


def extract_51d_features_from_noise_rgb(noise_rgb: np.ndarray) -> np.ndarray:
    if noise_rgb.ndim != 3 or noise_rgb.shape[2] != 3:
        raise ValueError("Expected noise_rgb with shape (H, W, 3)")

    per_channel: list[np.ndarray] = []
    row_avgs: list[np.ndarray] = []
    col_avgs: list[np.ndarray] = []

    for channel_index in range(3):
        channel_noise = noise_rgb[:, :, channel_index]
        per_channel.append(extract_15d_features_from_channel_noise(channel_noise))
        row_avgs.append(np.mean(channel_noise, axis=0))
        col_avgs.append(np.mean(channel_noise, axis=1))

    cross_row = np.array(
        [
            _safe_corrcoef(row_avgs[0], row_avgs[1]),
            _safe_corrcoef(row_avgs[0], row_avgs[2]),
            _safe_corrcoef(row_avgs[1], row_avgs[2]),
        ],
        dtype=np.float64,
    )
    cross_col = np.array(
        [
            _safe_corrcoef(col_avgs[0], col_avgs[1]),
            _safe_corrcoef(col_avgs[0], col_avgs[2]),
            _safe_corrcoef(col_avgs[1], col_avgs[2]),
        ],
        dtype=np.float64,
    )

    return np.concatenate([*per_channel, cross_row, cross_col], axis=0)
