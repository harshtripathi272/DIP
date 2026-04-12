from __future__ import annotations

import numpy as np

from .denoise import FilterbankConfig, apply_filterbank_rgb
from .noise_features import extract_51d_features_from_noise_rgb


def extract_204d_features_from_image(
    image_rgb: np.ndarray,
    config: FilterbankConfig | None = None,
) -> tuple[np.ndarray, list[str]]:
    denoised_variants = apply_filterbank_rgb(image_rgb, config=config)

    feature_vectors: list[np.ndarray] = []
    filter_names = list(denoised_variants.keys())

    for filter_name in filter_names:
        denoised = denoised_variants[filter_name]
        noise_rgb = image_rgb.astype(np.float64, copy=False) - denoised.astype(np.float64, copy=False)
        feature_vectors.append(extract_51d_features_from_noise_rgb(noise_rgb))

    return np.concatenate(feature_vectors, axis=0), filter_names
