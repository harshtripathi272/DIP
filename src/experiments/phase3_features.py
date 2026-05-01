from __future__ import annotations

import csv
from pathlib import Path
import sys

import numpy as np
from PIL import Image


if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config import DATASET_INDEX_PATH, PROCESSED_DATA_DIR, RAW_DATA_DIR, ExperimentConfig
from src.features.pipeline import extract_204d_features_from_image
from src.paths import ensure_project_directories

FEATURES_204D_PATH = PROCESSED_DATA_DIR / "features_204d.csv"
FEATURES_META_PATH = PROCESSED_DATA_DIR / "features_204d.meta.txt"


def _load_image_as_rgb_float(image_path: Path) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    array = np.asarray(image, dtype=np.float64) / 255.0
    return array


def _read_phase2_index() -> list[dict[str, str]]:
    if not DATASET_INDEX_PATH.exists():
        return []
    with DATASET_INDEX_PATH.open("r", newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        return list(reader)


def _feature_header() -> list[str]:
    cols = [
        "relative_path",
        "scanner_id",
        "file_format",
        "dpi",
        "scan_location",
        "jpeg_quality",
        "split",
    ]
    cols.extend([f"f{i}" for i in range(1, 205)])
    return cols


def main() -> None:
    ensure_project_directories()
    rows = _read_phase2_index()

    if not rows:
        FEATURES_204D_PATH.parent.mkdir(parents=True, exist_ok=True)
        with FEATURES_204D_PATH.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow(_feature_header())
        FEATURES_META_PATH.write_text("No samples found in dataset_index.csv\n", encoding="utf-8")
        print("No records found in dataset index. Created empty features_204d.csv header.")
        return

    written = 0
    skipped = 0
    active_filters: list[str] | None = None

    FEATURES_204D_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FEATURES_204D_PATH.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(_feature_header())

        for row in rows:
            rel_path = (row.get("relative_path") or "").strip()
            if not rel_path:
                skipped += 1
                continue

            image_path = RAW_DATA_DIR / rel_path
            if not image_path.exists() or not image_path.is_file():
                skipped += 1
                continue

            try:
                image_rgb = _load_image_as_rgb_float(image_path)
                
                config = ExperimentConfig()
                h, w = image_rgb.shape[:2]
                bh, bw = config.block_height, config.block_width
                
                if h > bh and w > bw:
                    blocks = []
                    for r in range(0, h - bh + 1, bh):
                        for c in range(0, w - bw + 1, bw):
                            blocks.append((r, c, image_rgb[r:r+bh, c:c+bw]))
                    if not blocks:
                        blocks = [(0, 0, image_rgb)]
                else:
                    blocks = [(0, 0, image_rgb)]
                
                for block_r, block_c, block_rgb in blocks:
                    features_204d, filter_names = extract_204d_features_from_image(block_rgb)

                    if features_204d.shape[0] != 204:
                        continue

                    if active_filters is None:
                        active_filters = filter_names

                    writer.writerow(
                        [
                            rel_path,
                            row.get("scanner_id", ""),
                            row.get("file_format", ""),
                            row.get("dpi", ""),
                            row.get("scan_location", ""),
                            row.get("jpeg_quality", ""),
                            row.get("split", ""),
                            *[f"{value:.10f}" for value in features_204d],
                        ]
                    )
                    written += 1
            except Exception:
                skipped += 1
                continue

    meta_lines = [
        f"records_in_index={len(rows)}",
        f"records_written={written}",
        f"records_skipped={skipped}",
        f"feature_dim=204",
    ]
    if active_filters:
        meta_lines.append(f"filters={','.join(active_filters)}")

    FEATURES_META_PATH.write_text("\n".join(meta_lines) + "\n", encoding="utf-8")

    print(f"Input records: {len(rows)}")
    print(f"Features written: {written}")
    print(f"Skipped records: {skipped}")
    print(f"Feature file: {FEATURES_204D_PATH}")


if __name__ == "__main__":
    main()
