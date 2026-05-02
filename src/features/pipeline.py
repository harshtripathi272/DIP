from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .denoise import FilterbankConfig, apply_filterbank_rgb
from .noise_features import extract_51d_features


def _load_image_as_rgb_float32(image_path: str | Path) -> np.ndarray:
    path = Path(image_path)
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        return np.asarray(rgb, dtype=np.float32) / np.float32(255.0)


def extract_204d_features(image_path: str | Path) -> np.ndarray:
    image_rgb = _load_image_as_rgb_float32(image_path)
    filter_outputs = apply_filterbank_rgb(image_rgb, config=FilterbankConfig(use_bm3d=True))

    feature_blocks: list[np.ndarray] = []
    for filter_name, denoised_rgb in filter_outputs:
        if filter_name not in {"lpa_ici", "median3", "wiener3", "wiener5"}:
            continue
        feature_blocks.append(extract_51d_features(image_rgb, denoised_rgb))

    if len(feature_blocks) != 4:
        raise RuntimeError("Expected four 51-D blocks from the denoising filterbank")

    features = np.concatenate(feature_blocks, axis=0).astype(np.float32, copy=False)
    if features.shape != (204,):
        raise RuntimeError(f"Expected a 204-D feature vector, got {features.shape}")
    return features


def extract_204d_features_from_image(
    image_rgb: np.ndarray,
    config: FilterbankConfig | None = None,
) -> tuple[np.ndarray, list[str]]:
    image_rgb = np.asarray(image_rgb, dtype=np.float32)
    filter_outputs = apply_filterbank_rgb(image_rgb, config=config or FilterbankConfig())
    feature_blocks: list[np.ndarray] = []
    filter_names: list[str] = []

    for filter_name, denoised_rgb in filter_outputs:
        feature_blocks.append(extract_51d_features(image_rgb, denoised_rgb))
        filter_names.append(filter_name)

    features = np.concatenate(feature_blocks, axis=0).astype(np.float32, copy=False)
    return features, filter_names