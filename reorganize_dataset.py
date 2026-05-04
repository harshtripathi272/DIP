from __future__ import annotations

import csv
import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split

from src.config import DATASET_CSV_PATH, PROJECT_ROOT, RAW_DATA_DIR, STANDARDIZED_DATA_DIR


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
TARGET_MAX_DIMENSION = 1024
RANDOM_SEED = 42


@dataclass(frozen=True)
class DatasetItem:
    source_path: Path
    scanner_label: int


@dataclass(frozen=True)
class SourceSpec:
    label: int
    root: Path
    name: str


def _collect_images(root_dir: Path, scanner_label: int) -> list[DatasetItem]:
    if not root_dir.exists() or not root_dir.is_dir():
        return []

    items: list[DatasetItem] = []
    for path in sorted(root_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            items.append(DatasetItem(source_path=path, scanner_label=scanner_label))
    return items


def _standardize_image(source_path: Path, destination_root: Path, source_root: Path) -> Path:
    relative_path = source_path.relative_to(source_root).with_suffix(".png")
    destination_path = destination_root / relative_path
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(source_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((TARGET_MAX_DIMENSION, TARGET_MAX_DIMENSION), Image.Resampling.LANCZOS)
        image.save(destination_path, format="PNG")

    return destination_path


def _write_dataset_csv(records: list[dict[str, str]]) -> None:
    DATASET_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DATASET_CSV_PATH.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["image_path", "scanner_label", "split"])
        writer.writeheader()
        writer.writerows(records)


def _parse_source_specs(source_args: list[str]) -> list[SourceSpec]:
    specs: list[SourceSpec] = []
    for index, raw_value in enumerate(source_args):
        root = Path(raw_value).expanduser().resolve()
        specs.append(SourceSpec(label=index, root=root, name=root.name or f"source_{index}"))
    return specs


def _resolve_named_source_specs(dataset_root: Path) -> list[SourceSpec]:
    expected_names = [
        "s1_epson_4490",
        "s2_hp_scanjet_6300c_1_SG9CO270W5",
        "s3_hp_scanjet_6300c_2",
        "s4_hp_scanjet_8250",
    ]

    specs: list[SourceSpec] = []
    for index, folder_name in enumerate(expected_names):
        root = dataset_root / folder_name
        specs.append(SourceSpec(label=index, root=root, name=folder_name))
    return specs


def _default_source_specs() -> list[SourceSpec]:
    return [
        SourceSpec(label=0, root=RAW_DATA_DIR / "footprints" / "Dactyloscopic", name="Dactyloscopic"),
        SourceSpec(label=1, root=RAW_DATA_DIR / "footprints" / "Scanned", name="Scanned"),
    ]


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Standardize images, assign numeric labels, and create dataset.csv for the classifier."
        )
    )
    parser.add_argument(
        "--dataset-root",
        metavar="PATH",
        help=(
            "Parent folder that contains the four scanner datasets named s1_epson_4490, s2_hp_scanjet_6300c_1_SG9CO270W5, s3_hp_scanjet_6300c_2, and s4_hp_scanjet_8250."
        ),
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="PATH",
        help=(
            "Path to a source dataset folder. Repeat this flag once per class, in the order you want labels 0..N-1 assigned."
        ),
    )
    return parser


def main() -> None:
    parser = _build_argument_parser()
    args = parser.parse_args()

    if args.dataset_root:
        dataset_root = Path(args.dataset_root).expanduser().resolve()
        if not dataset_root.exists() or not dataset_root.is_dir():
            raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
        source_specs = _resolve_named_source_specs(dataset_root)
    else:
        if not RAW_DATA_DIR.exists() or not RAW_DATA_DIR.is_dir():
            raise FileNotFoundError(f"Raw data directory not found: {RAW_DATA_DIR}")
        source_specs = _parse_source_specs(args.source) if args.source else _default_source_specs()

    items: list[DatasetItem] = []
    label_to_root: dict[int, Path] = {}
    label_to_name: dict[int, str] = {}

    for spec in source_specs:
        if not spec.root.exists() or not spec.root.is_dir():
            raise FileNotFoundError(f"Expected dataset folder not found: {spec.root}")
        label_to_root[spec.label] = spec.root
        label_to_name[spec.label] = spec.name
        items.extend(_collect_images(spec.root, spec.label))

    if not items:
        source_list = ", ".join(str(spec.root) for spec in source_specs)
        raise FileNotFoundError(f"No images found under the provided source folders: {source_list}")

    labels = np.array([item.scanner_label for item in items], dtype=np.int32)
    indices = np.arange(len(items), dtype=np.int32)

    train_indices, test_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=labels,
    )

    split_lookup = {int(index): "train" for index in train_indices}
    split_lookup.update({int(index): "test" for index in test_indices})

    standardized_root = STANDARDIZED_DATA_DIR
    records: list[dict[str, str]] = []

    for index, item in enumerate(items):
        source_root = label_to_root[item.scanner_label]
        standardized_path = _standardize_image(item.source_path, standardized_root, source_root)
        records.append(
            {
                "image_path": standardized_path.relative_to(PROJECT_ROOT).as_posix(),
                "scanner_label": str(item.scanner_label),
                "split": split_lookup[index],
            }
        )

    _write_dataset_csv(records)

    train_count = sum(1 for record in records if record["split"] == "train")
    test_count = sum(1 for record in records if record["split"] == "test")

    print(f"Total images: {len(records)}")
    print(f"Train split: {train_count}")
    print(f"Test split: {test_count}")
    print(f"Dataset CSV: {DATASET_CSV_PATH}")
    print(f"Standardized images: {standardized_root}")
    print("Label mapping:")
    for label in sorted(label_to_name):
        print(f"  {label}: {label_to_name[label]} -> {label_to_root[label]}")


if __name__ == "__main__":
    main()