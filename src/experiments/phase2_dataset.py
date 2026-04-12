from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable


if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config import (
    DATASET_INDEX_PATH,
    DATASET_MANIFEST_PATH,
    DATASET_MANIFEST_TEMPLATE_PATH,
    DATASET_PLACEHOLDER_ROOT,
    DATASET_SUMMARY_PATH,
    RAW_DATA_DIR,
    TEST_SPLIT_PATH,
    TRAIN_SPLIT_PATH,
    ExperimentConfig,
)
from src.paths import ensure_project_directories

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
CSV_COLUMNS = [
    "relative_path",
    "scanner_id",
    "file_format",
    "dpi",
    "scan_location",
    "jpeg_quality",
    "split",
]


@dataclass(frozen=True)
class SampleRecord:
    relative_path: str
    scanner_id: str
    file_format: str
    dpi: str
    scan_location: str
    jpeg_quality: str

    def to_row(self, split: str) -> dict[str, str]:
        return {
            "relative_path": self.relative_path,
            "scanner_id": self.scanner_id,
            "file_format": self.file_format,
            "dpi": self.dpi,
            "scan_location": self.scan_location,
            "jpeg_quality": self.jpeg_quality,
            "split": split,
        }


def ensure_manifest_template() -> None:
    if not DATASET_MANIFEST_TEMPLATE_PATH.exists():
        DATASET_MANIFEST_TEMPLATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with DATASET_MANIFEST_TEMPLATE_PATH.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(
                fp,
                fieldnames=[
                    "relative_path",
                    "scanner_id",
                    "file_format",
                    "dpi",
                    "scan_location",
                    "jpeg_quality",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "relative_path": "PLACEHOLDER_DATASET_ROOT/S1/TIFF/200/loc1/example_001.tif",
                    "scanner_id": "S1",
                    "file_format": "TIFF",
                    "dpi": "200",
                    "scan_location": "loc1",
                    "jpeg_quality": "",
                }
            )


def ensure_placeholder_paths() -> None:
    DATASET_PLACEHOLDER_ROOT.mkdir(parents=True, exist_ok=True)
    placeholder_sample_dir = DATASET_PLACEHOLDER_ROOT / "S1" / "TIFF" / "200" / "loc1"
    placeholder_sample_dir.mkdir(parents=True, exist_ok=True)


def infer_record_from_relative_path(relative_path: str) -> SampleRecord:
    path = Path(relative_path)
    parts = list(path.parts)

    scanner_id = parts[1] if len(parts) > 1 else "UNKNOWN_SCANNER"
    file_format = parts[2] if len(parts) > 2 else path.suffix.lstrip(".").upper() or "UNKNOWN_FORMAT"
    dpi = parts[3] if len(parts) > 3 else "UNKNOWN_DPI"
    scan_location = parts[4] if len(parts) > 4 else "UNKNOWN_LOCATION"

    jpeg_quality = ""
    lower_name = path.stem.lower()
    if "q=" in lower_name:
        jpeg_quality = lower_name.split("q=")[-1].split("_")[0]

    return SampleRecord(
        relative_path=relative_path.replace("\\", "/"),
        scanner_id=scanner_id,
        file_format=file_format,
        dpi=dpi,
        scan_location=scan_location,
        jpeg_quality=jpeg_quality,
    )


def load_records_from_manifest() -> list[SampleRecord]:
    if not DATASET_MANIFEST_PATH.exists():
        return []

    records: list[SampleRecord] = []
    with DATASET_MANIFEST_PATH.open("r", newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            relative_path = (row.get("relative_path") or "").strip()
            if not relative_path:
                continue
            record = SampleRecord(
                relative_path=relative_path.replace("\\", "/"),
                scanner_id=(row.get("scanner_id") or "UNKNOWN_SCANNER").strip(),
                file_format=(row.get("file_format") or "UNKNOWN_FORMAT").strip(),
                dpi=(row.get("dpi") or "UNKNOWN_DPI").strip(),
                scan_location=(row.get("scan_location") or "UNKNOWN_LOCATION").strip(),
                jpeg_quality=(row.get("jpeg_quality") or "").strip(),
            )
            records.append(record)
    return records


def load_records_from_directory() -> list[SampleRecord]:
    records: list[SampleRecord] = []
    for image_path in RAW_DATA_DIR.rglob("*"):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        try:
            relative_path = image_path.relative_to(RAW_DATA_DIR).as_posix()
        except ValueError:
            continue

        # Skip generated artifacts and placeholders.
        if relative_path.startswith("dataset_manifest"):
            continue

        records.append(infer_record_from_relative_path(relative_path))
    return records


def deduplicate_records(records: Iterable[SampleRecord]) -> list[SampleRecord]:
    dedup: dict[str, SampleRecord] = {}
    for record in records:
        dedup[record.relative_path] = record
    return sorted(dedup.values(), key=lambda item: item.relative_path)


def assign_splits(records: list[SampleRecord], train_split: float, seed: int) -> dict[str, str]:
    by_scanner: dict[str, list[SampleRecord]] = defaultdict(list)
    for record in records:
        by_scanner[record.scanner_id].append(record)

    rng = random.Random(seed)
    split_map: dict[str, str] = {}

    for scanner_id, scanner_records in by_scanner.items():
        scanner_records = sorted(scanner_records, key=lambda item: item.relative_path)
        rng.shuffle(scanner_records)

        if len(scanner_records) == 1:
            split_map[scanner_records[0].relative_path] = "train"
            continue

        train_count = int(round(len(scanner_records) * train_split))
        train_count = max(1, min(train_count, len(scanner_records) - 1))

        for index, record in enumerate(scanner_records):
            split_map[record.relative_path] = "train" if index < train_count else "test"

    return split_map


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows: list[dict[str, str]]) -> None:
    by_scanner: dict[str, int] = defaultdict(int)
    by_split: dict[str, int] = defaultdict(int)

    for row in rows:
        by_scanner[row["scanner_id"]] += 1
        by_split[row["split"]] += 1

    summary = {
        "total_samples": len(rows),
        "scanner_counts": dict(sorted(by_scanner.items())),
        "split_counts": dict(sorted(by_split.items())),
        "paths": {
            "dataset_index": str(DATASET_INDEX_PATH),
            "train_split": str(TRAIN_SPLIT_PATH),
            "test_split": str(TEST_SPLIT_PATH),
            "manifest": str(DATASET_MANIFEST_PATH),
            "manifest_template": str(DATASET_MANIFEST_TEMPLATE_PATH),
        },
    }

    DATASET_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATASET_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    cfg = ExperimentConfig()
    ensure_project_directories()
    ensure_placeholder_paths()
    ensure_manifest_template()

    manifest_records = load_records_from_manifest()
    discovered_records = load_records_from_directory()

    records = deduplicate_records([*manifest_records, *discovered_records])
    split_map = assign_splits(records, cfg.train_split, cfg.random_seed)

    rows = [record.to_row(split_map.get(record.relative_path, "train")) for record in records]

    write_csv(DATASET_INDEX_PATH, rows)
    write_csv(TRAIN_SPLIT_PATH, [row for row in rows if row["split"] == "train"])
    write_csv(TEST_SPLIT_PATH, [row for row in rows if row["split"] == "test"])
    write_summary(rows)

    print(f"Indexed samples: {len(rows)}")
    print(f"Manifest path (optional): {DATASET_MANIFEST_PATH}")
    print(f"Manifest template: {DATASET_MANIFEST_TEMPLATE_PATH}")
    print(f"Dataset index: {DATASET_INDEX_PATH}")
    print(f"Train split: {TRAIN_SPLIT_PATH}")
    print(f"Test split: {TEST_SPLIT_PATH}")


if __name__ == "__main__":
    main()
