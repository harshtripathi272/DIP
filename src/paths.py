from pathlib import Path

from .config import DATA_DIR, FIGURES_DIR, MODELS_DIR, OUTPUTS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, REPORTS_DIR


DIRECTORIES = (
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    OUTPUTS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    MODELS_DIR,
)


def ensure_project_directories() -> list[Path]:
    created_paths: list[Path] = []
    for directory in DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)
        created_paths.append(directory)
    return created_paths
