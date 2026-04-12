from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = PROCESSED_DATA_DIR / "splits"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

DATASET_PLACEHOLDER_ROOT = RAW_DATA_DIR / "PLACEHOLDER_DATASET_ROOT"
DATASET_MANIFEST_PATH = RAW_DATA_DIR / "dataset_manifest.csv"
DATASET_MANIFEST_TEMPLATE_PATH = RAW_DATA_DIR / "dataset_manifest.template.csv"
DATASET_INDEX_PATH = PROCESSED_DATA_DIR / "dataset_index.csv"
TRAIN_SPLIT_PATH = SPLITS_DIR / "train.csv"
TEST_SPLIT_PATH = SPLITS_DIR / "test.csv"
DATASET_SUMMARY_PATH = PROCESSED_DATA_DIR / "dataset_summary.json"


@dataclass(frozen=True)
class ExperimentConfig:
    random_seed: int = 42
    block_height: int = 1024
    block_width: int = 768
    train_split: float = 0.8
    jpeg_quality_levels: tuple[int, ...] = (70, 80, 90)
    svm_c_grid: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
    svm_gamma_grid: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0)
