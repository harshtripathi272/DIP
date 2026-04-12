from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import wiener


FilterFn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class FilterbankConfig:
    use_bm3d_if_available: bool = True


def _clip01(channel: np.ndarray) -> np.ndarray:
    return np.clip(channel, 0.0, 1.0)


def _denoise_bm3d_or_fallback(channel: np.ndarray) -> np.ndarray:
    try:
        from bm3d import bm3d  # type: ignore

        denoised = bm3d(channel, sigma_psd=0.05)
        return _clip01(denoised.astype(np.float64, copy=False))
    except Exception:
        # Fallback keeps phase-3 runnable even without bm3d installed.
        return _clip01(wiener(channel, mysize=5).astype(np.float64, copy=False))


def _denoise_median3(channel: np.ndarray) -> np.ndarray:
    return _clip01(median_filter(channel, size=3).astype(np.float64, copy=False))


def _denoise_wiener3(channel: np.ndarray) -> np.ndarray:
    return _clip01(wiener(channel, mysize=3).astype(np.float64, copy=False))


def _denoise_wiener5(channel: np.ndarray) -> np.ndarray:
    return _clip01(wiener(channel, mysize=5).astype(np.float64, copy=False))


def get_filterbank(config: FilterbankConfig | None = None) -> dict[str, FilterFn]:
    cfg = config or FilterbankConfig()
    filterbank: dict[str, FilterFn] = {
        "median3": _denoise_median3,
        "wiener3": _denoise_wiener3,
        "wiener5": _denoise_wiener5,
    }
    if cfg.use_bm3d_if_available:
        filterbank["bm3d_or_fallback"] = _denoise_bm3d_or_fallback
    else:
        filterbank["wiener5_alt"] = _denoise_wiener5
    return filterbank


def apply_filterbank_rgb(
    image_rgb: np.ndarray,
    config: FilterbankConfig | None = None,
) -> dict[str, np.ndarray]:
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected image_rgb with shape (H, W, 3)")

    image_rgb = image_rgb.astype(np.float64, copy=False)
    filters = get_filterbank(config)
    outputs: dict[str, np.ndarray] = {}

    for filter_name, filter_fn in filters.items():
        denoised = np.zeros_like(image_rgb, dtype=np.float64)
        for channel_index in range(3):
            denoised[:, :, channel_index] = filter_fn(image_rgb[:, :, channel_index])
        outputs[filter_name] = denoised

    return outputs
