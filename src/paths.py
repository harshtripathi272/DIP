from pathlib import Path

from .config import (
    DATASET_PLACEHOLDER_ROOT,
    DATA_DIR,
    RESULTS_DIR,
    FIGURES_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    REPORTS_DIR,
    SPLITS_DIR,
    STANDARDIZED_DATA_DIR,
)


DIRECTORIES = (
    DATA_DIR,
    RAW_DATA_DIR,
    DATASET_PLACEHOLDER_ROOT,
    PROCESSED_DATA_DIR,
    SPLITS_DIR,
    OUTPUTS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    STANDARDIZED_DATA_DIR,
)


def ensure_project_directories() -> list[Path]:
    created_paths: list[Path] = []
    for directory in DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)
        created_paths.append(directory)
    return created_paths
