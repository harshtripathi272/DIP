from __future__ import annotations

import csv
from pathlib import Path
import sys

import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import json


if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config import (
    DATASET_INDEX_PATH,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    ExperimentConfig,
)
from src.paths import ensure_project_directories

FEATURES_CSV_PATH = PROCESSED_DATA_DIR / "features_204d.csv"
MODEL_PIPELINE_PATH = MODELS_DIR / "classifier_pipeline.pkl"
RESULTS_JSON_PATH = PROCESSED_DATA_DIR / "phase4_results.json"
CONFUSION_MATRIX_PATH = PROCESSED_DATA_DIR / "confusion_matrix.csv"
CLASSIFICATION_REPORT_PATH = PROCESSED_DATA_DIR / "classification_report.txt"


def _load_features_and_labels() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray]:
    if not FEATURES_CSV_PATH.exists():
        raise FileNotFoundError(f"Features file not found: {FEATURES_CSV_PATH}")

    x_list: list[list[float]] = []
    y_list: list[str] = []
    split_list: list[str] = []
    path_list: list[str] = []

    with FEATURES_CSV_PATH.open("r", newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            scanner_id = (row.get("scanner_id") or "").strip()
            split = (row.get("split") or "").strip()
            rel_path = (row.get("relative_path") or "").strip()

            feature_values: list[float] = []
            for col_name in row:
                if col_name.startswith("f") and col_name[1:].isdigit():
                    try:
                        feature_values.append(float(row[col_name]))
                    except ValueError:
                        pass

            if not feature_values:
                continue

            x_list.append(feature_values)
            y_list.append(scanner_id)
            split_list.append(split)
            path_list.append(rel_path)

    if not x_list:
        raise ValueError("No valid feature rows found in features CSV")

    x = np.array(x_list, dtype=np.float64)
    y_unique = sorted(set(y_list))
    y_encoded = np.array([y_unique.index(label) for label in y_list], dtype=np.int32)
    split_array = np.array(split_list, dtype=str)
    path_array = np.array(path_list, dtype=str)

    return x, y_encoded, split_array, y_unique, path_array


def _build_pipeline_with_grid_search(
    x_train: np.ndarray,
    y_train: np.ndarray,
    cfg: ExperimentConfig,
) -> tuple[Pipeline, dict]:
    train_classes, train_counts = np.unique(y_train, return_counts=True)
    if train_classes.shape[0] < 2:
        raise ValueError("Need at least two scanner classes in train split for classification")

    min_class_count = int(np.min(train_counts))
    if min_class_count < 2:
        raise ValueError("Each scanner class needs at least 2 train samples for StratifiedKFold")

    n_lda_components = min(10, x_train.shape[1] - 1, train_classes.shape[0] - 1)
    n_cv_splits = min(3, min_class_count)

    param_grid = {
        "svm__C": cfg.svm_c_grid,
        "svm__gamma": cfg.svm_gamma_grid,
    }

    base_pipeline = Pipeline(
        [
            ("scaler", MinMaxScaler(feature_range=(-1, 1))),
            ("lda", LinearDiscriminantAnalysis(n_components=n_lda_components)),
            ("svm", SVC(kernel="rbf", random_state=cfg.random_seed, verbose=0)),
        ]
    )

    grid_search = GridSearchCV(
        base_pipeline,
        param_grid,
        cv=StratifiedKFold(n_splits=n_cv_splits, shuffle=True, random_state=cfg.random_seed),
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(x_train, y_train)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    return grid_search.best_estimator_, {"best_params": best_params, "best_cv_score": float(best_score)}


def main() -> None:
    cfg = ExperimentConfig()
    ensure_project_directories()

    print("Loading features and labels...")
    try:
        x, y_encoded, split_array, y_unique, path_array = _load_features_and_labels()
    except ValueError as err:
        print(f"No feature data available: {err}")
        print("\nPhase 4 requires Phase 3 outputs (features_204d.csv).")
        print("To generate features, you must:")
        print("  1. Add scanner images to data/raw/")
        print("  2. Update data/raw/dataset_manifest.csv with paths and metadata")
        print("  3. Run: python src/experiments/phase2_dataset.py")
        print("  4. Run: python src/experiments/phase3_features.py")
        print("  5. Run: python src/experiments/phase4_classification.py")
        return

    print(f"Total samples: {len(x)}")
    print(f"Feature dimension: {x.shape[1]}")
    print(f"Scanner classes: {len(y_unique)}")
    print(f"Scanners: {y_unique}")

    train_mask = split_array == "train"
    test_mask = split_array == "test"

    x_train = x[train_mask]
    y_train = y_encoded[train_mask]
    x_test = x[test_mask]
    y_test = y_encoded[test_mask]
    path_test = path_array[test_mask]

    print(f"\nTrain set size: {len(x_train)}")
    print(f"Test set size: {len(x_test)}")

    if len(x_train) == 0 or len(x_test) == 0:
        raise ValueError("Train or test set is empty after splitting")

    print("\nRunning grid search for best SVM hyperparameters...")
    best_pipeline, grid_info = _build_pipeline_with_grid_search(x_train, y_train, cfg)

    print(f"Best CV score: {grid_info['best_cv_score']:.4f}")
    print(f"Best params: {grid_info['best_params']}")

    print("\nEvaluating on test set...")
    y_pred_blocks = best_pipeline.predict(x_test)
    
    # Majority voting code
    unique_paths = np.unique(path_test)
    y_test_agg = []
    y_pred_agg = []
    
    for path in unique_paths:
        path_mask = path_test == path
        y_test_path = y_test[path_mask]
        
        # Ground truth is same for all blocks of the image
        y_test_agg.append(y_test_path[0])
        
        # Majority voting
        preds = y_pred_blocks[path_mask]
        values, counts = np.unique(preds, return_counts=True)
        majority_pred = values[np.argmax(counts)]
        y_pred_agg.append(majority_pred)
        
    y_test_agg = np.array(y_test_agg)
    y_pred_agg = np.array(y_pred_agg)

    test_accuracy = accuracy_score(y_test_agg, y_pred_agg)
    print(f"Test accuracy (Image level): {test_accuracy:.4f}")

    cm = confusion_matrix(y_test_agg, y_pred_agg)
    report = classification_report(y_test_agg, y_pred_agg, target_names=y_unique, zero_division=0)

    MODEL_PIPELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(best_pipeline, MODEL_PIPELINE_PATH)
    print(f"Model saved to: {MODEL_PIPELINE_PATH}")

    results = {
        "test_accuracy": float(test_accuracy),
        "train_set_size": len(x_train),
        "test_set_size": len(x_test),
        "feature_dimension": x.shape[1],
        "n_classes": len(y_unique),
        "scanner_classes": y_unique,
        "grid_search": grid_info,
        "confusion_matrix": cm.tolist(),
    }

    RESULTS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_JSON_PATH.open("w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)
    print(f"Results saved to: {RESULTS_JSON_PATH}")

    with CONFUSION_MATRIX_PATH.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow([""] + y_unique)
        for i, scanner_id in enumerate(y_unique):
            writer.writerow([scanner_id] + cm[i].tolist())
    print(f"Confusion matrix saved to: {CONFUSION_MATRIX_PATH}")

    with CLASSIFICATION_REPORT_PATH.open("w", encoding="utf-8") as fp:
        fp.write(report)
    print(f"Classification report saved to: {CLASSIFICATION_REPORT_PATH}")

    print("\n" + "=" * 60)
    print("Classification Report:")
    print("=" * 60)
    print(report)


if __name__ == "__main__":
    main()
