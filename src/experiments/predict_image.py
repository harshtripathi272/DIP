from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from PIL import Image


if __package__ is None or __package__ == "":
	sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, ExperimentConfig
from src.features.pipeline import extract_204d_features_from_image


MODEL_PIPELINE_PATH = MODELS_DIR / "classifier_pipeline.pkl"
RESULTS_JSON_PATH = PROCESSED_DATA_DIR / "phase4_results.json"


def _load_image_as_rgb_float(image_path: Path) -> np.ndarray:
	image = Image.open(image_path).convert("RGB")
	return np.asarray(image, dtype=np.float64) / 255.0


def _load_scanner_labels() -> list[str]:
	if not RESULTS_JSON_PATH.exists():
		raise FileNotFoundError(
			f"Label map not found at {RESULTS_JSON_PATH}. Run phase4 classification first."
		)

	with RESULTS_JSON_PATH.open("r", encoding="utf-8") as fp:
		results = json.load(fp)

	labels = results.get("scanner_classes")
	if not isinstance(labels, list) or not labels:
		raise ValueError(f"scanner_classes missing or invalid in {RESULTS_JSON_PATH}")

	return [str(label) for label in labels]


def _extract_blocks(image_rgb: np.ndarray, cfg: ExperimentConfig) -> list[np.ndarray]:
	h, w = image_rgb.shape[:2]
	block_height = cfg.block_height
	block_width = cfg.block_width

	if h >= block_height and w >= block_width:
		blocks: list[np.ndarray] = []
		for row in range(0, h - block_height + 1, block_height):
			for col in range(0, w - block_width + 1, block_width):
				blocks.append(image_rgb[row : row + block_height, col : col + block_width])
		if blocks:
			return blocks

	return [image_rgb]


def _majority_vote(predictions: np.ndarray) -> tuple[int, dict[int, int]]:
	values, counts = np.unique(predictions, return_counts=True)
	count_map = {int(value): int(count) for value, count in zip(values, counts, strict=True)}
	winner_index = int(values[np.argmax(counts)])
	return winner_index, count_map


def predict_single_image(image_path: Path) -> None:
	if not MODEL_PIPELINE_PATH.exists():
		raise FileNotFoundError(
			f"Model pipeline not found at {MODEL_PIPELINE_PATH}. Run phase4 classification first."
		)

	labels = _load_scanner_labels()
	model = joblib.load(MODEL_PIPELINE_PATH)
	cfg = ExperimentConfig()
	image_rgb = _load_image_as_rgb_float(image_path)
	blocks = _extract_blocks(image_rgb, cfg)

	feature_rows: list[np.ndarray] = []
	for block in blocks:
		features_204d, _ = extract_204d_features_from_image(block)
		if features_204d.shape[0] != 204:
			raise ValueError(f"Unexpected feature dimension {features_204d.shape[0]} for {image_path}")
		feature_rows.append(features_204d)

	feature_matrix = np.vstack(feature_rows)
	block_predictions = np.asarray(model.predict(feature_matrix), dtype=np.int32)
	majority_index, vote_counts = _majority_vote(block_predictions)

	print(f"Image: {image_path}")
	print(f"Blocks: {len(blocks)}")
	print(f"Predicted scanner index: {majority_index}")
	print(f"Predicted scanner label: {labels[majority_index]}")
	print(f"Block votes: {vote_counts}")

	if len(block_predictions) > 1:
		decoded_predictions = [labels[int(pred)] for pred in block_predictions]
		print(f"Block predictions: {decoded_predictions}")


def main() -> None:
	parser = argparse.ArgumentParser(description="Predict the scanner for a single image")
	parser.add_argument("image_path", type=Path, help="Path to the image to classify")
	args = parser.parse_args()

	if not args.image_path.exists():
		raise FileNotFoundError(f"Image not found: {args.image_path}")

	predict_single_image(args.image_path)


if __name__ == "__main__":
	main()