from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import wiener


FilterFn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class FilterbankConfig:
    use_bm3d: bool = True


def _ensure_float32(channel: np.ndarray) -> np.ndarray:
    array = np.asarray(channel, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("Expected a 2D single-channel image")
    return np.clip(array, 0.0, 1.0).astype(np.float32, copy=False)


def _finalize(channel: np.ndarray) -> np.ndarray:
    return np.clip(np.nan_to_num(channel, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0).astype(np.float32, copy=False)


def denoise_lpa_ici(channel: np.ndarray) -> np.ndarray:
    channel = _ensure_float32(channel)
    try:
        from bm3d import bm3d
    except ImportError as exc:
        raise ImportError("bm3d is required for the LPA-ICI filter. Install it with `pip install bm3d`.") from exc

    denoised = bm3d(channel, sigma_psd=0.02)
    return _finalize(np.asarray(denoised, dtype=np.float32))


def denoise_median3(channel: np.ndarray) -> np.ndarray:
    channel = _ensure_float32(channel)
    return _finalize(median_filter(channel, size=3).astype(np.float32, copy=False))


def denoise_wiener3(channel: np.ndarray) -> np.ndarray:
    channel = _ensure_float32(channel)
    return _finalize(wiener(channel, mysize=3).astype(np.float32, copy=False))


def denoise_wiener5(channel: np.ndarray) -> np.ndarray:
    channel = _ensure_float32(channel)
    return _finalize(wiener(channel, mysize=5).astype(np.float32, copy=False))


def get_filterbank(config: FilterbankConfig | None = None) -> list[tuple[str, FilterFn]]:
    cfg = config or FilterbankConfig()
    filterbank: list[tuple[str, FilterFn]] = []
    if cfg.use_bm3d:
        filterbank.append(("lpa_ici", denoise_lpa_ici))
    filterbank.extend(
        [
            ("median3", denoise_median3),
            ("wiener3", denoise_wiener3),
            ("wiener5", denoise_wiener5),
        ]
    )
    return filterbank


def apply_filterbank_rgb(
    image_rgb: np.ndarray,
    config: FilterbankConfig | None = None,
) -> list[tuple[str, np.ndarray]]:
    image_rgb = np.asarray(image_rgb, dtype=np.float32)
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected image_rgb with shape (H, W, 3)")

    outputs: list[tuple[str, np.ndarray]] = []
    for filter_name, filter_fn in get_filterbank(config):
        denoised = np.empty_like(image_rgb, dtype=np.float32)
        for channel_index in range(3):
            denoised[:, :, channel_index] = filter_fn(image_rgb[:, :, channel_index])
        outputs.append((filter_name, denoised.astype(np.float32, copy=False)))
    return outputs