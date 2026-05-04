from __future__ import annotations

import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC


if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config import FEATURES_CSV_PATH, PROJECT_ROOT, RESULTS_DIR, ExperimentConfig
from src.paths import ensure_project_directories


REPORT_PATH = RESULTS_DIR / "report.txt"
CONFUSION_MATRIX_PNG_PATH = RESULTS_DIR / "confusion_matrix.png"


def _load_features() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    if not FEATURES_CSV_PATH.exists():
        raise FileNotFoundError(f"Features file not found: {FEATURES_CSV_PATH}")

    feature_rows: list[list[float]] = []
    labels: list[int] = []
    splits: list[str] = []

    with FEATURES_CSV_PATH.open("r", newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            image_path = (row.get("image_path") or "").strip()
            scanner_label = (row.get("scanner_label") or "").strip()
            split = (row.get("split") or "").strip()

            if not image_path or not scanner_label:
                continue

            values: list[float] = []
            for index in range(204):
                value = row.get(f"f{index}")
                if value is None or value == "":
                    values = []
                    break
                values.append(float(value))

            if len(values) != 204:
                continue

            feature_rows.append(values)
            labels.append(int(scanner_label))
            splits.append(split)

    if not feature_rows:
        raise ValueError("No valid feature rows found in features.csv")

    x = np.asarray(feature_rows, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32)
    split_array = np.asarray(splits, dtype=str)
    class_names = [str(label) for label in sorted(set(labels))]
    return x, y, split_array, class_names


def _fit_scaler_and_lda(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    cfg: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray, MinMaxScaler, LinearDiscriminantAnalysis]:
    scaler = MinMaxScaler(feature_range=(-1, 1))
    x_train_scaled = scaler.fit_transform(x_train).astype(np.float32, copy=False)
    x_test_scaled = scaler.transform(x_test).astype(np.float32, copy=False)

    n_classes = len(np.unique(y_train))
    n_features = x_train_scaled.shape[1]
    n_components = min(10, n_classes - 1, n_features - 1)
    if n_components < 1:
        raise ValueError("LDA needs at least two classes and two features")

    lda = LinearDiscriminantAnalysis(n_components=n_components)
    x_train_lda = lda.fit_transform(x_train_scaled, y_train).astype(np.float32, copy=False)
    x_test_lda = lda.transform(x_test_scaled).astype(np.float32, copy=False)
    return x_train_lda, x_test_lda, scaler, lda


def _train_svm(x_train: np.ndarray, y_train: np.ndarray, cfg: ExperimentConfig) -> GridSearchCV:
    param_grid = {
        "C": [0.1, 1.0, 10.0, 100.0],
        "gamma": ["scale", 0.01, 0.1, 1.0],
    }
    svm = SVC(kernel="rbf")
    class_counts = np.bincount(y_train)
    min_class_count = int(class_counts[class_counts > 0].min())
    n_splits = min(5, min_class_count)
    if n_splits < 2:
        raise ValueError("Need at least two training samples per class for SVM cross-validation")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.random_seed)
    grid_search = GridSearchCV(svm, param_grid=param_grid, cv=cv, n_jobs=-1, verbose=1)
    grid_search.fit(x_train, y_train)
    return grid_search


def _save_confusion_matrix_png(matrix: np.ndarray, class_names: list[str]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")

    threshold = matrix.max() / 2.0 if matrix.size else 0.0
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            ax.text(
                col_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(CONFUSION_MATRIX_PNG_PATH, dpi=200)
    plt.close(fig)


def main() -> None:
    cfg = ExperimentConfig()
    ensure_project_directories()

    x, y, split_array, class_names = _load_features()
    train_mask = split_array == "train"
    test_mask = split_array == "test"

    x_train = x[train_mask]
    y_train = y[train_mask]
    x_test = x[test_mask]
    y_test = y[test_mask]

    if x_train.size == 0 or x_test.size == 0:
        raise ValueError("Train or test split is empty")

    x_train_lda, x_test_lda, _, _ = _fit_scaler_and_lda(x_train, y_train, x_test, cfg)
    grid_search = _train_svm(x_train_lda, y_train, cfg)

    best_svm = grid_search.best_estimator_
    y_pred = best_svm.predict(x_test_lda)

    labels = [int(label) for label in sorted(set(y.tolist()))]
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    overall_accuracy = accuracy_score(y_test, y_pred)
    per_class_accuracy = np.divide(
        np.diag(cm),
        cm.sum(axis=1),
        out=np.zeros(cm.shape[0], dtype=np.float32),
        where=cm.sum(axis=1) != 0,
    )
    report = classification_report(y_test, y_pred, labels=labels, target_names=class_names, zero_division=0)

    print("Confusion matrix:")
    print(cm)
    for class_name, class_accuracy in zip(class_names, per_class_accuracy, strict=True):
        print(f"Class {class_name} accuracy: {class_accuracy:.4f}")
    print(f"Overall accuracy: {overall_accuracy:.4f}")
    print(f"Best SVM params: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")

    _save_confusion_matrix_png(cm, class_names)
    REPORT_PATH.write_text(report, encoding="utf-8")

    print(f"Confusion matrix saved to: {CONFUSION_MATRIX_PNG_PATH}")
    print(f"Classification report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()