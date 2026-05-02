from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.stats import kurtosis, skew


def _as_float32(array: np.ndarray) -> np.ndarray:
    return np.asarray(array, dtype=np.float32)


def extract_noise(image_channel: np.ndarray, denoised_channel: np.ndarray) -> np.ndarray:
    return _as_float32(image_channel) - _as_float32(denoised_channel)


def _safe_corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    a = _as_float32(a)
    b = _as_float32(b)
    if a.size == 0 or b.size == 0:
        return 0.0
    if a.size < 2 or b.size < 2:
        return 0.0
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return 0.0
    value = np.corrcoef(a, b)[0, 1]
    if not np.isfinite(value):
        return 0.0
    return float(value)


def _safe_skew(x: np.ndarray) -> float:
    x = _as_float32(x)
    if x.size < 3 or np.allclose(x, x[0]):
        return 0.0
    value = skew(x, bias=False, nan_policy="omit")
    if not np.isfinite(value):
        return 0.0
    return float(value)


def _safe_kurtosis(x: np.ndarray) -> float:
    x = _as_float32(x)
    if x.size < 4 or np.allclose(x, x[0]):
        return 0.0
    value = kurtosis(x, fisher=True, bias=False, nan_policy="omit")
    if not np.isfinite(value):
        return 0.0
    return float(value)


def _channel_features(noise: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if noise.ndim != 2:
        raise ValueError("Expected a 2D noise map for one channel")

    noise = _as_float32(noise)
    m, n = noise.shape
    if m < 2 or n < 2:
        raise ValueError("Need at least a 2x2 noise map for feature extraction")

    row_avg = np.mean(noise, axis=0, dtype=np.float32)
    col_avg = np.mean(noise, axis=1, dtype=np.float32)

    rho_row = np.array([_safe_corrcoef(row_avg, noise[row_idx, :]) for row_idx in range(m)], dtype=np.float32)
    rho_col = np.array([_safe_corrcoef(col_avg, noise[:, col_idx]) for col_idx in range(n)], dtype=np.float32)

    f1 = float(np.mean(rho_row, dtype=np.float32))
    f2 = float(np.std(rho_row, dtype=np.float32))
    f3 = _safe_skew(rho_row)
    f4 = _safe_kurtosis(rho_row)

    f5 = float(np.mean(rho_col, dtype=np.float32))
    f6 = float(np.std(rho_col, dtype=np.float32))
    f7 = _safe_skew(rho_col)
    f8 = _safe_kurtosis(rho_col)

    f9 = float(np.std(row_avg, dtype=np.float32))
    f10 = _safe_skew(row_avg)
    f11 = _safe_kurtosis(row_avg)

    f12 = float(np.std(col_avg, dtype=np.float32))
    f13 = _safe_skew(col_avg)
    f14 = _safe_kurtosis(col_avg)

    if abs(f1) < 1e-12:
        f15 = 0.0
    else:
        f15 = float((1.0 - (f5 / f1)) * 100.0)

    features = np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15], dtype=np.float32)
    return features, row_avg.astype(np.float32, copy=False), col_avg.astype(np.float32, copy=False), rho_row.astype(np.float32, copy=False)


def extract_51d_features(image_rgb: np.ndarray, denoised_rgb: np.ndarray) -> np.ndarray:
    image_rgb = _as_float32(image_rgb)
    denoised_rgb = _as_float32(denoised_rgb)

    if image_rgb.ndim != 3 or denoised_rgb.ndim != 3 or image_rgb.shape != denoised_rgb.shape or image_rgb.shape[2] != 3:
        raise ValueError("Expected image_rgb and denoised_rgb with shape (H, W, 3)")

    channel_features: list[np.ndarray] = []
    row_avgs: list[np.ndarray] = []
    col_avgs: list[np.ndarray] = []

    for channel_index in range(3):
        noise = image_rgb[:, :, channel_index] - denoised_rgb[:, :, channel_index]
        features_15, row_avg, col_avg, _ = _channel_features(noise)
        channel_features.append(features_15)
        row_avgs.append(row_avg)
        col_avgs.append(col_avg)

    cross_features: list[float] = []
    for first_index, second_index in combinations(range(3), 2):
        cross_features.append(_safe_corrcoef(row_avgs[first_index], row_avgs[second_index]))
        cross_features.append(_safe_corrcoef(col_avgs[first_index], col_avgs[second_index]))

    block = np.concatenate(
        [
            channel_features[0],
            channel_features[1],
            channel_features[2],
            np.asarray(cross_features, dtype=np.float32),
        ],
        axis=0,
    )
    return block.astype(np.float32, copy=False)


def extract_51d_features_from_noise_rgb(noise_rgb: np.ndarray) -> np.ndarray:
    noise_rgb = _as_float32(noise_rgb)
    if noise_rgb.ndim != 3 or noise_rgb.shape[2] != 3:
        raise ValueError("Expected noise_rgb with shape (H, W, 3)")
    zeros = np.zeros_like(noise_rgb, dtype=np.float32)
    return extract_51d_features(noise_rgb, zeros)