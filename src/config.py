from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"


@dataclass(frozen=True)
class ExperimentConfig:
    random_seed: int = 42
    block_height: int = 1024
    block_width: int = 768
    train_split: float = 0.8
    jpeg_quality_levels: tuple[int, ...] = (70, 80, 90)
    svm_c_grid: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
    svm_gamma_grid: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0)
