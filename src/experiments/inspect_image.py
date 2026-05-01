from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image


if __package__ is None or __package__ == "":
	sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config import ExperimentConfig
from src.features.denoise import FilterbankConfig, apply_filterbank_rgb
from src.features.pipeline import extract_204d_features_from_image


def _load_image_as_rgb_float(image_path: Path) -> np.ndarray:
	image = Image.open(image_path).convert("RGB")
	return np.asarray(image, dtype=np.float64) / 255.0


def _residual_to_display(residual_rgb: np.ndarray) -> np.ndarray:
	# Collapse RGB residuals into a single magnitude map for easier inspection.
	return np.mean(np.abs(residual_rgb), axis=2)


def inspect_image(image_path: Path, output_path: Path | None = None) -> None:
	try:
		import matplotlib.pyplot as plt
	except Exception as exc:
		raise RuntimeError(
			"matplotlib is required for visualization. Run this inside the project venv or install the plotting dependencies."
		) from exc

	cfg = ExperimentConfig()
	image_rgb = _load_image_as_rgb_float(image_path)
	denoised_variants = apply_filterbank_rgb(image_rgb, config=FilterbankConfig())
	features_204d, filter_names = extract_204d_features_from_image(image_rgb)

	print(f"Image: {image_path}")
	print(f"Image shape: {image_rgb.shape}")
	print(f"Active filters: {', '.join(filter_names)}")
	print(f"Feature vector shape: {features_204d.shape}")
	print("First 15 features:")
	print(np.array2string(features_204d[:15], precision=6, separator=", "))

	columns = 1 + len(denoised_variants)
	fig, axes = plt.subplots(2, columns, figsize=(4 * columns, 8), constrained_layout=True)
	if columns == 1:
		axes = np.array([[axes[0]], [axes[1]]])

	axes[0, 0].imshow(image_rgb)
	axes[0, 0].set_title("Original")
	axes[0, 0].axis("off")

	axes[1, 0].imshow(_residual_to_display(image_rgb - np.mean(image_rgb, axis=2, keepdims=True)), cmap="gray")
	axes[1, 0].set_title("Original luminance residual")
	axes[1, 0].axis("off")

	for idx, (filter_name, denoised_rgb) in enumerate(denoised_variants.items(), start=1):
		residual_rgb = image_rgb - denoised_rgb
		residual_map = _residual_to_display(residual_rgb)

		axes[0, idx].imshow(np.clip(denoised_rgb, 0.0, 1.0))
		axes[0, idx].set_title(f"Denoised: {filter_name}")
		axes[0, idx].axis("off")

		axes[1, idx].imshow(residual_map, cmap="magma")
		axes[1, idx].set_title(f"Residual: {filter_name}")
		axes[1, idx].axis("off")

	fig.suptitle(
		f"Noise inspection | block size {cfg.block_height}x{cfg.block_width} only matters for large images",
		fontsize=12,
	)

	if output_path is not None:
		output_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(output_path, dpi=200)
		print(f"Saved visualization to: {output_path}")

	plt.show()


def main() -> None:
	parser = argparse.ArgumentParser(description="Inspect scanner noise extraction for one image")
	parser.add_argument("image_path", type=Path, help="Path to the image to inspect")
	parser.add_argument(
		"--output",
		type=Path,
		default=None,
		help="Optional path to save the visualization PNG",
	)
	args = parser.parse_args()

	if not args.image_path.exists():
		raise FileNotFoundError(f"Image not found: {args.image_path}")

	inspect_image(args.image_path, args.output)


if __name__ == "__main__":
	main()