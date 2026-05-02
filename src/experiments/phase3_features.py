from __future__ import annotations

import csv
from pathlib import Path
import sys

from tqdm import tqdm


if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config import DATASET_CSV_PATH, FEATURES_CSV_PATH, PROJECT_ROOT, RESULTS_DIR
from src.features.pipeline import extract_204d_features
from src.paths import ensure_project_directories


FEATURE_NAMES = [f"f{i}" for i in range(204)]
FEATURE_LOG_PATH = RESULTS_DIR / "phase3_feature_errors.log"


def _read_dataset_rows() -> list[dict[str, str]]:
    if not DATASET_CSV_PATH.exists():
        return []
    with DATASET_CSV_PATH.open("r", newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def _resolve_image_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _write_feature_header() -> None:
    FEATURES_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FEATURES_CSV_PATH.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["image_path", "scanner_label", "split", *FEATURE_NAMES])


def main() -> None:
    ensure_project_directories()
    rows = _read_dataset_rows()
    _write_feature_header()

    if not rows:
        print(f"No records found in {DATASET_CSV_PATH}. Created empty features.csv header.")
        return

    FEATURES_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEATURE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    failures: list[str] = []

    with FEATURES_CSV_PATH.open("a", newline="", encoding="utf-8") as features_fp:
        writer = csv.writer(features_fp)
        for row in tqdm(rows, desc="Extracting features", unit="image"):
            image_path_raw = (row.get("image_path") or "").strip()
            scanner_label = (row.get("scanner_label") or "").strip()
            split = (row.get("split") or "").strip()

            if not image_path_raw:
                skipped += 1
                failures.append("missing image_path")
                continue

            image_path = _resolve_image_path(image_path_raw)
            if not image_path.exists():
                skipped += 1
                failures.append(f"missing file: {image_path_raw}")
                continue

            try:
                features = extract_204d_features(image_path)
                if features.shape != (204,):
                    raise ValueError(f"Unexpected feature shape: {features.shape}")
                writer.writerow([image_path_raw, scanner_label, split, *[f"{value:.10f}" for value in features]])
                written += 1
            except Exception as exc:
                skipped += 1
                failures.append(f"{image_path_raw}: {exc}")

    if failures:
        FEATURE_LOG_PATH.write_text("\n".join(failures) + "\n", encoding="utf-8")
    else:
        FEATURE_LOG_PATH.write_text("", encoding="utf-8")

    print(f"Input records: {len(rows)}")
    print(f"Features written: {written}")
    print(f"Skipped records: {skipped}")
    print(f"Feature file: {FEATURES_CSV_PATH}")
    print(f"Failure log: {FEATURE_LOG_PATH}")


if __name__ == "__main__":
    main()