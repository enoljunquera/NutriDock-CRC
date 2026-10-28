#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Food-level dietary baseline for NutriDock-CRC reviewer response.

Purpose
-------
Evaluate the complete food-level dietary representation (307 food variables)
without chemical-exposure, molecular-descriptor or docking-derived features.

The script performs:

1) Primary reviewer experiment:
   Three-class (Normal / Polyp / Neoplasia) 5-fold stratified OOF baseline
   using the same three model families used in the manuscript baseline ablation:
   - Multinomial logistic regression
   - Linear SVM
   - Random Forest

2) Robustness experiment:
   Repeated stratified 5-fold CV (20 repetitions) for the same three-class task.

3) Main binary endpoint:
   Repeated stratified 5-fold CV (20 repetitions) for Normal vs Altered,
   with training-side-only threshold calibration:
   - Youden threshold
   - Fixed specificity target = 0.90
   AUROC and AUPRC are computed from continuous held-out scores.

Important
---------
Until the clinical/nutrition team confirms that these 307 variables are the
processed food-level representation derived from the 155-item FFQ, describe
this experiment as a "food-level dietary baseline" rather than literally
"the 155-item FFQ baseline".

No target-driven feature selection is performed. All preprocessing is fitted
inside the training folds to prevent leakage.

Dependencies
------------
pandas
numpy
scikit-learn

Run
---
python food_baseline_reviewer.py \
    --data dataset_baseline_alimentos.csv \
    --outdir food_baseline_results

Optional:
python food_baseline_reviewer.py \
    --data dataset_baseline_alimentos.csv \
    --outdir food_baseline_results \
    --repetitions 20 \
    --seed 42
"""

from __future__ import annotations

import argparse
import json
import platform
import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import sklearn

from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

ID_COL = "Codigo_dieta"
DIAG_COL = "Diagnostico"
Y3_COL = "y_3class"
GROUP_BINARY_COL = "Grupo_binario"
YBIN_COL = "y_binary"

NON_FEATURE_COLS = {
    ID_COL,
    DIAG_COL,
    Y3_COL,
    GROUP_BINARY_COL,
    YBIN_COL,
}

EXPECTED_N = 111
EXPECTED_FEATURES = 307

THREE_CLASS_MAP = {
    0: "Normal",
    1: "Polyp",
    2: "Neoplasia",
}

SPECIFICITY_TARGET = 0.90


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Food-level baseline experiments for NutriDock-CRC."
    )
    parser.add_argument(
        "--data",
        default="dataset_baseline_alimentos.csv",
        help="Input CSV created from the dietary database.",
    )
    parser.add_argument(
        "--outdir",
        default="food_baseline_results",
        help="Directory where results will be written.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=20,
        help="Number of repetitions for repeated 5-fold evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Master random seed.",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> Tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(path)

    required = {ID_COL, DIAG_COL, Y3_COL, GROUP_BINARY_COL, YBIN_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df[ID_COL].duplicated().any():
        duplicated = df.loc[df[ID_COL].duplicated(), ID_COL].tolist()
        raise ValueError(f"Duplicated subject IDs detected: {duplicated}")

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]

    # Numeric conversion is explicit so accidental text cells do not pass silently.
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if len(df) != EXPECTED_N:
        warnings.warn(
            f"Expected {EXPECTED_N} subjects, found {len(df)}. "
            "The script will continue, but verify the cohort."
        )

    if len(feature_cols) != EXPECTED_FEATURES:
        warnings.warn(
            f"Expected {EXPECTED_FEATURES} food variables, found {len(feature_cols)}. "
            "The script will continue, but verify the input."
        )

    if df[feature_cols].isna().any().any():
        n_missing = int(df[feature_cols].isna().sum().sum())
        warnings.warn(
            f"{n_missing} missing predictor values detected. "
            "They will be median-imputed INSIDE training folds."
        )

    # Basic label checks.
    if not set(df[Y3_COL].dropna().unique()).issubset({0, 1, 2}):
        raise ValueError(f"Unexpected labels in {Y3_COL}.")
    if not set(df[YBIN_COL].dropna().unique()).issubset({0, 1}):
        raise ValueError(f"Unexpected labels in {YBIN_COL}.")

    return df, feature_cols


def make_models(seed: int) -> Dict[str, Pipeline]:
    """
    Fixed, conservative baseline models.

    Linear models:
      median imputation -> standardization -> L2-regularized classifier

    Random Forest:
      median imputation -> RF

    No feature selection is performed.
    """
    logistic = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                C=1.0,
                penalty="l2",
                solver="lbfgs",
                class_weight="balanced",
                max_iter=10000,
                random_state=seed,
            ),
        ),
    ])

    linear_svm = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        (
            "model",
            LinearSVC(
                C=1.0,
                class_weight="balanced",
                dual="auto",
                max_iter=50000,
                random_state=seed,
            ),
        ),
    ])

    random_forest = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        (
            "model",
            RandomForestClassifier(
                n_estimators=500,
                class_weight="balanced_subsample",
                max_features="sqrt",
                min_samples_leaf=1,
                random_state=seed,
                n_jobs=-1,
            ),
        ),
    ])

    return {
        "Multinomial LR": logistic,
        "Linear SVM": linear_svm,
        "Random Forest": random_forest,
    }


def get_binary_score(model, X: pd.DataFrame) -> np.ndarray:
    """
    Return a continuous score where larger values indicate Altered (class 1).
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        classes = np.asarray(model.classes_)
        pos_idx = int(np.where(classes == 1)[0][0])
        return np.asarray(proba[:, pos_idx], dtype=float)

    if hasattr(model, "decision_function"):
        score = np.asarray(model.decision_function(X), dtype=float)
        if score.ndim != 1:
            raise ValueError("Binary decision_function returned a non-1D score.")
        return score

    raise TypeError("Model has neither predict_proba nor decision_function.")


def choose_youden_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    j = tpr - fpr
    best = int(np.nanargmax(j))
    return float(thresholds[best])


def specificity_at_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> float:
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return float(tn / (tn + fp)) if (tn + fp) else np.nan


def choose_fixed_specificity_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    target: float = SPECIFICITY_TARGET,
) -> float:
    """
    Choose the most permissive threshold that still reaches target specificity
    on training-side calibration predictions.

    More permissive = the LOWEST threshold satisfying specificity >= target.
    """
    unique_scores = np.unique(scores)

    # Include a threshold above the maximum score to guarantee specificity=1.
    eps = np.finfo(float).eps * max(1.0, abs(float(np.max(scores))))
    candidates = np.concatenate([
        unique_scores,
        [float(np.max(scores)) + eps],
    ])

    valid = []
    for threshold in candidates:
        spec = specificity_at_threshold(y_true, scores, float(threshold))
        if np.isfinite(spec) and spec >= target:
            valid.append(float(threshold))

    if not valid:
        # Defensive fallback; practically unreachable because threshold > max(scores)
        # predicts every subject as Normal.
        return float(np.max(scores) + 1.0)

    return float(min(valid))


def binary_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict:
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan

    return {
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "accuracy": accuracy_score(y_true, pred),
        "f1_altered": f1_score(y_true, pred, pos_label=1, zero_division=0),
        "sensitivity_altered": sensitivity,
        "specificity_normal": specificity,
        "auroc": roc_auc_score(y_true, scores),
        "auprc": average_precision_score(y_true, scores),
        "threshold": threshold,
        "n_test": len(y_true),
        "n_true_normal": int((y_true == 0).sum()),
        "n_true_altered": int((y_true == 1).sum()),
        "n_pred_normal": int((pred == 0).sum()),
        "n_pred_altered": int((pred == 1).sum()),
    }


def three_class_metrics(y_true: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "macro_f1": f1_score(y_true, pred, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "accuracy": accuracy_score(y_true, pred),
    }


# ---------------------------------------------------------------------
# Experiment 1: manuscript-compatible 5-fold OOF three-class baseline
# ---------------------------------------------------------------------

def run_single_5fold_threeclass(
    df: pd.DataFrame,
    feature_cols: list[str],
    outdir: Path,
    seed: int,
) -> pd.DataFrame:
    X = df[feature_cols]
    y = df[Y3_COL].astype(int).to_numpy()
    ids = df[ID_COL].astype(str).to_numpy()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    all_summaries = []
    all_fold_rows = []
    all_pred_rows = []

    for model_name, base_model in make_models(seed).items():
        oof_pred = np.full(len(df), -1, dtype=int)
        fold_macro_f1 = []

        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
            model = clone(base_model)
            model.fit(X.iloc[train_idx], y[train_idx])
            pred = model.predict(X.iloc[test_idx]).astype(int)

            oof_pred[test_idx] = pred
            met = three_class_metrics(y[test_idx], pred)
            fold_macro_f1.append(met["macro_f1"])

            all_fold_rows.append({
                "experiment": "single_5fold_threeclass",
                "model": model_name,
                "fold": fold,
                **met,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
            })

            for idx, p in zip(test_idx, pred):
                all_pred_rows.append({
                    ID_COL: ids[idx],
                    "model": model_name,
                    "fold": fold,
                    "y_true": int(y[idx]),
                    "y_pred": int(p),
                    "y_true_label": THREE_CLASS_MAP[int(y[idx])],
                    "y_pred_label": THREE_CLASS_MAP[int(p)],
                })

        if np.any(oof_pred < 0):
            raise RuntimeError(f"OOF predictions incomplete for {model_name}.")

        oof = three_class_metrics(y, oof_pred)

        all_summaries.append({
            "representation": "Food-level dietary baseline",
            "model": model_name,
            "n_features": len(feature_cols),
            "oof_macro_f1": oof["macro_f1"],
            "oof_balanced_accuracy": oof["balanced_accuracy"],
            "oof_accuracy": oof["accuracy"],
            "fold_macro_f1_mean": float(np.mean(fold_macro_f1)),
            "fold_macro_f1_sd": float(np.std(fold_macro_f1, ddof=1)),
        })

    summary_df = pd.DataFrame(all_summaries)
    pd.DataFrame(all_fold_rows).to_csv(
        outdir / "01_threeclass_single5fold_fold_metrics.csv", index=False
    )
    pd.DataFrame(all_pred_rows).to_csv(
        outdir / "01_threeclass_single5fold_oof_predictions.csv", index=False
    )
    summary_df.to_csv(
        outdir / "01_threeclass_single5fold_summary.csv", index=False
    )

    return summary_df


# ---------------------------------------------------------------------
# Experiment 2: repeated 20 x 5 three-class robustness
# ---------------------------------------------------------------------

def run_repeated_threeclass(
    df: pd.DataFrame,
    feature_cols: list[str],
    outdir: Path,
    master_seed: int,
    repetitions: int,
) -> pd.DataFrame:
    X = df[feature_cols]
    y = df[Y3_COL].astype(int).to_numpy()

    fold_rows = []

    for rep in range(1, repetitions + 1):
        rep_seed = master_seed + rep - 1
        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=rep_seed,
        )

        models = make_models(rep_seed)

        for model_name, base_model in models.items():
            for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
                model = clone(base_model)
                model.fit(X.iloc[train_idx], y[train_idx])
                pred = model.predict(X.iloc[test_idx]).astype(int)

                met = three_class_metrics(y[test_idx], pred)

                # Neoplasia-oriented descriptive metrics.
                y_neo = (y[test_idx] == 2).astype(int)
                pred_neo = (pred == 2).astype(int)
                tn, fp, fn, tp = confusion_matrix(
                    y_neo, pred_neo, labels=[0, 1]
                ).ravel()

                neo_precision = precision_score(
                    y_neo, pred_neo, zero_division=0
                )
                neo_recall = recall_score(
                    y_neo, pred_neo, zero_division=0
                )
                neo_specificity = tn / (tn + fp) if (tn + fp) else np.nan
                neo_hit = int(tp >= 1)

                fold_rows.append({
                    "representation": "Food-level dietary baseline",
                    "model": model_name,
                    "repetition": rep,
                    "fold": fold,
                    **met,
                    "neo_precision": neo_precision,
                    "neo_recall": neo_recall,
                    "neo_specificity": neo_specificity,
                    "neo_hit": neo_hit,
                    "true_neoplasia": int(y_neo.sum()),
                    "predicted_neoplasia": int(pred_neo.sum()),
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                })

    folds = pd.DataFrame(fold_rows)
    folds.to_csv(
        outdir / "02_threeclass_repeated20x5_fold_metrics.csv",
        index=False,
    )

    metric_cols = [
        "macro_f1",
        "balanced_accuracy",
        "accuracy",
        "neo_precision",
        "neo_recall",
        "neo_specificity",
        "neo_hit",
        "true_neoplasia",
        "predicted_neoplasia",
    ]

    summary_rows = []
    for model_name, group in folds.groupby("model"):
        row = {
            "representation": "Food-level dietary baseline",
            "model": model_name,
            "n_features": len(feature_cols),
            "n_repetitions": repetitions,
            "n_fold_evaluations": len(group),
        }
        for metric in metric_cols:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_sd"] = group[metric].std(ddof=1)
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        outdir / "02_threeclass_repeated20x5_summary.csv",
        index=False,
    )

    # Repetition-level means are useful for later paired tests if Combined is
    # re-run on exactly the same splits.
    repetition_means = (
        folds.groupby(["model", "repetition"], as_index=False)[
            ["macro_f1", "balanced_accuracy", "accuracy"]
        ]
        .mean()
    )
    repetition_means.to_csv(
        outdir / "02_threeclass_repetition_means.csv",
        index=False,
    )

    return summary


# ---------------------------------------------------------------------
# Experiment 3: repeated binary baseline with leakage-safe thresholds
# ---------------------------------------------------------------------

def generate_training_side_scores(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    base_model,
    seed: int,
) -> np.ndarray:
    """
    Produce OOF scores using ONLY the current outer-training set.

    These scores are used solely for threshold selection, so the outer-test fold
    remains completely untouched.
    """
    inner_cv = StratifiedKFold(
        n_splits=4,
        shuffle=True,
        random_state=seed,
    )

    calibration_scores = np.full(len(y_train), np.nan, dtype=float)

    for inner_train_idx, inner_val_idx in inner_cv.split(X_train, y_train):
        model = clone(base_model)
        model.fit(
            X_train.iloc[inner_train_idx],
            y_train[inner_train_idx],
        )
        calibration_scores[inner_val_idx] = get_binary_score(
            model,
            X_train.iloc[inner_val_idx],
        )

    if np.isnan(calibration_scores).any():
        raise RuntimeError("Incomplete training-side calibration scores.")

    return calibration_scores


def run_repeated_binary(
    df: pd.DataFrame,
    feature_cols: list[str],
    outdir: Path,
    master_seed: int,
    repetitions: int,
) -> pd.DataFrame:
    X = df[feature_cols]
    y = df[YBIN_COL].astype(int).to_numpy()
    ids = df[ID_COL].astype(str).to_numpy()

    fold_rows = []
    prediction_rows = []

    for rep in range(1, repetitions + 1):
        rep_seed = master_seed + rep - 1
        outer_cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=rep_seed,
        )

        models = make_models(rep_seed)

        for model_name, base_model in models.items():
            for fold, (train_idx, test_idx) in enumerate(
                outer_cv.split(X, y),
                start=1,
            ):
                X_train = X.iloc[train_idx]
                X_test = X.iloc[test_idx]
                y_train = y[train_idx]
                y_test = y[test_idx]

                # ---------------------------------------------------------
                # Training-side-only threshold calibration
                # ---------------------------------------------------------
                calibration_scores = generate_training_side_scores(
                    X_train=X_train,
                    y_train=y_train,
                    base_model=base_model,
                    seed=rep_seed + (fold * 1000),
                )

                threshold_youden = choose_youden_threshold(
                    y_train,
                    calibration_scores,
                )

                threshold_spec90 = choose_fixed_specificity_threshold(
                    y_train,
                    calibration_scores,
                    target=SPECIFICITY_TARGET,
                )

                # ---------------------------------------------------------
                # Final fit on full outer-training fold
                # ---------------------------------------------------------
                final_model = clone(base_model)
                final_model.fit(X_train, y_train)
                test_scores = get_binary_score(final_model, X_test)

                # Threshold-independent metrics are identical under both
                # threshold policies; we retain them in both rows for a
                # manuscript-friendly table.
                for policy, threshold in [
                    ("Youden", threshold_youden),
                    ("Fixed specificity 0.90", threshold_spec90),
                ]:
                    met = binary_metrics(
                        y_true=y_test,
                        scores=test_scores,
                        threshold=threshold,
                    )

                    fold_rows.append({
                        "representation": "Food-level dietary baseline",
                        "model": model_name,
                        "threshold_policy": policy,
                        "repetition": rep,
                        "fold": fold,
                        **met,
                    })

                # Save continuous OOF scores once per outer test fold.
                # Threshold-specific predictions are also included.
                pred_youden = (test_scores >= threshold_youden).astype(int)
                pred_spec90 = (test_scores >= threshold_spec90).astype(int)

                for local_pos, original_idx in enumerate(test_idx):
                    prediction_rows.append({
                        ID_COL: ids[original_idx],
                        "model": model_name,
                        "repetition": rep,
                        "fold": fold,
                        "y_true": int(y_test[local_pos]),
                        "score_altered": float(test_scores[local_pos]),
                        "threshold_youden": float(threshold_youden),
                        "pred_youden": int(pred_youden[local_pos]),
                        "threshold_spec90": float(threshold_spec90),
                        "pred_spec90": int(pred_spec90[local_pos]),
                    })

    folds = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(prediction_rows)

    folds.to_csv(
        outdir / "03_binary_repeated20x5_fold_metrics.csv",
        index=False,
    )
    predictions.to_csv(
        outdir / "03_binary_repeated20x5_oof_predictions.csv",
        index=False,
    )

    metric_cols = [
        "balanced_accuracy",
        "accuracy",
        "f1_altered",
        "sensitivity_altered",
        "specificity_normal",
        "auroc",
        "auprc",
        "threshold",
    ]

    summary_rows = []
    for (model_name, policy), group in folds.groupby(
        ["model", "threshold_policy"]
    ):
        row = {
            "representation": "Food-level dietary baseline",
            "model": model_name,
            "threshold_policy": policy,
            "n_features": len(feature_cols),
            "n_repetitions": repetitions,
            "n_fold_evaluations": len(group),
        }
        for metric in metric_cols:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_sd"] = group[metric].std(ddof=1)
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        outdir / "03_binary_repeated20x5_summary.csv",
        index=False,
    )

    # Repetition-level means for later paired comparison with Combined,
    # provided Combined is re-run using exactly the same outer splits.
    repetition_means = (
        folds.groupby(
            ["model", "threshold_policy", "repetition"],
            as_index=False,
        )[
            [
                "balanced_accuracy",
                "accuracy",
                "f1_altered",
                "sensitivity_altered",
                "specificity_normal",
                "auroc",
                "auprc",
            ]
        ]
        .mean()
    )

    repetition_means.to_csv(
        outdir / "03_binary_repetition_means.csv",
        index=False,
    )

    return summary


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------

def write_dataset_summary(
    df: pd.DataFrame,
    feature_cols: list[str],
    outdir: Path,
    args,
):
    summary = {
        "input_file": str(Path(args.data).resolve()),
        "n_subjects": int(len(df)),
        "n_features_food_level": int(len(feature_cols)),
        "three_class_counts": {
            str(k): int(v)
            for k, v in df[DIAG_COL].value_counts().to_dict().items()
        },
        "binary_counts": {
            str(k): int(v)
            for k, v in df[GROUP_BINARY_COL].value_counts().to_dict().items()
        },
        "missing_predictor_values": int(
            df[feature_cols].isna().sum().sum()
        ),
        "master_seed": int(args.seed),
        "repetitions": int(args.repetitions),
        "outer_cv": "StratifiedKFold(n_splits=5, shuffle=True)",
        "binary_threshold_calibration": (
            "4-fold stratified OOF predictions within outer-training data only"
        ),
        "specificity_target": SPECIFICITY_TARGET,
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__,
    }

    with open(outdir / "00_run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    pd.DataFrame({
        "feature": feature_cols,
        "missing_n": [int(df[c].isna().sum()) for c in feature_cols],
        "zero_fraction": [
            float((df[c].fillna(0) == 0).mean()) for c in feature_cols
        ],
        "mean": [float(df[c].mean()) for c in feature_cols],
        "sd": [float(df[c].std(ddof=1)) for c in feature_cols],
        "min": [float(df[c].min()) for c in feature_cols],
        "max": [float(df[c].max()) for c in feature_cols],
    }).to_csv(
        outdir / "00_food_feature_descriptives.csv",
        index=False,
    )


def print_summary_table(title: str, df: pd.DataFrame):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    with pd.option_context(
        "display.max_columns", None,
        "display.width", 220,
        "display.max_rows", 50,
    ):
        print(df.to_string(index=False))


def main():
    args = parse_args()
    data_path = Path(args.data)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"Input file not found: {data_path}")

    df, feature_cols = load_dataset(data_path)
    write_dataset_summary(df, feature_cols, outdir, args)

    print("\nDataset loaded successfully")
    print(f"Subjects: {len(df)}")
    print(f"Food-level predictors: {len(feature_cols)}")
    print("\nThree-class distribution:")
    print(df[DIAG_COL].value_counts())
    print("\nBinary distribution:")
    print(df[GROUP_BINARY_COL].value_counts())

    summary_5fold = run_single_5fold_threeclass(
        df=df,
        feature_cols=feature_cols,
        outdir=outdir,
        seed=args.seed,
    )
    print_summary_table(
        "PRIMARY REVIEWER EXPERIMENT: FOOD-LEVEL BASELINE, SINGLE 5-FOLD OOF",
        summary_5fold,
    )

    summary_rep3 = run_repeated_threeclass(
        df=df,
        feature_cols=feature_cols,
        outdir=outdir,
        master_seed=args.seed,
        repetitions=args.repetitions,
    )
    print_summary_table(
        f"ROBUSTNESS: THREE-CLASS FOOD BASELINE, {args.repetitions} x 5-FOLD",
        summary_rep3,
    )

    summary_bin = run_repeated_binary(
        df=df,
        feature_cols=feature_cols,
        outdir=outdir,
        master_seed=args.seed,
        repetitions=args.repetitions,
    )
    print_summary_table(
        f"BINARY FOOD BASELINE, {args.repetitions} x 5-FOLD",
        summary_bin,
    )

    print("\nDone.")
    print(f"Results written to: {outdir.resolve()}")
    print("\nFiles to inspect first:")
    print("  01_threeclass_single5fold_summary.csv")
    print("  02_threeclass_repeated20x5_summary.csv")
    print("  03_binary_repeated20x5_summary.csv")
    print("\nFor later paired comparisons, keep:")
    print("  02_threeclass_repetition_means.csv")
    print("  03_binary_repetition_means.csv")
    print("  03_binary_repeated20x5_oof_predictions.csv")


if __name__ == "__main__":
    main()
