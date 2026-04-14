# learning_curve_projection.py
# Estimate how binary screening performance scales with more patients
# using repeated stratified subsampling + 5-fold CV, without recomputing features.

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.metrics import (
    balanced_accuracy_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve,
)

import matplotlib.pyplot as plt


# -----------------------------
# Utilities
# -----------------------------

def infer_label_column(df: pd.DataFrame) -> str:
    for cand in ["label", "y", "target", "class", "Class", "LABEL"]:
        if cand in df.columns:
            return cand
    # Fallback: last column
    return df.columns[-1]


def numeric_feature_frame(df: pd.DataFrame, drop_cols: List[str]) -> pd.DataFrame:
    work = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    # Keep numeric
    num = work.select_dtypes(include=[np.number]).copy()
    # Fill missing
    for c in num.columns:
        if num[c].isna().any():
            num[c] = num[c].fillna(num[c].median())
    return num


def to_binary_labels(y_multi: np.ndarray) -> np.ndarray:
    # 0 = Normal, 1 = Polyp, 2 = Neoplasia
    # Binary: 0 = Normal, 1 = Altered (Polyp + Neoplasia)
    return (y_multi != 0).astype(int)


def specificity_from_cm(cm: np.ndarray) -> float:
    # cm for binary: [[tn, fp],[fn,tp]]
    tn, fp, fn, tp = cm.ravel()
    denom = tn + fp
    return float(tn) / float(denom) if denom > 0 else 0.0


def pick_threshold_youden(y_true: np.ndarray, p: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y_true, p)

    # Drop the 'inf' threshold returned by sklearn
    mask = np.isfinite(thr)
    fpr, tpr, thr = fpr[mask], tpr[mask], thr[mask]

    if thr.size == 0:
        return 0.5

    youden = tpr - fpr
    j = int(np.argmax(youden))
    t = float(thr[j])

    # Probabilities are in [0,1]
    return float(np.clip(t, 0.0, 1.0))


def pick_threshold_fixed_specificity(y_true: np.ndarray, p: np.ndarray, target_spec: float) -> float:
    fpr, tpr, thr = roc_curve(y_true, p)

    # Drop the 'inf' threshold
    mask = np.isfinite(thr)
    fpr, tpr, thr = fpr[mask], tpr[mask], thr[mask]

    if thr.size == 0:
        return 0.5

    spec = 1.0 - fpr
    ok = np.where(spec >= target_spec)[0]
    if ok.size == 0:
        # If nothing meets target specificity, return conservative threshold
        return 1.0

    best = ok[np.argmax(tpr[ok])]
    t = float(thr[best])
    return float(np.clip(t, 0.0, 1.0))


def fit_power_law(n: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Fit y(n) ~ a - b * n^{-alpha} by searching alpha and solving linear least squares for a,b.
    Returns dict with a,b,alpha,mse.
    """
    n = np.asarray(n, dtype=float)
    y = np.asarray(y, dtype=float)

    best = {"a": float(np.mean(y)), "b": 0.0, "alpha": 0.5, "mse": float("inf")}
    # Guard
    if len(n) < 3:
        best["mse"] = 0.0
        return best

    alphas = np.linspace(0.10, 1.20, 111)  # 0.10..1.20
    for alpha in alphas:
        x = n ** (-alpha)
        # y = a - b*x => y = a + c*x with c=-b
        A = np.vstack([np.ones_like(x), x]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        a = float(coef[0])
        c = float(coef[1])
        y_hat = a + c * x
        mse = float(np.mean((y_hat - y) ** 2))
        if mse < best["mse"]:
            best = {"a": a, "b": -c, "alpha": float(alpha), "mse": mse}
    return best


def estimate_n_for_target(params: Dict[str, float], target: float) -> Optional[float]:
    """
    Using s(n) = a - b*n^{-alpha}, solve for n needed to reach target.
    Returns None if not reachable (target >= a) or invalid.
    """
    a, b, alpha = params["a"], params["b"], params["alpha"]
    if b <= 0 or alpha <= 0:
        return None
    # target < a required
    if target >= a:
        return None
    denom = (a - target)
    if denom <= 0:
        return None
    n = (b / denom) ** (1.0 / alpha)
    if not np.isfinite(n) or n <= 0:
        return None
    return float(n)


# -----------------------------
# Models
# -----------------------------

def make_model(model_name: str, seed: int) -> object:
    if model_name == "mlp":
        # Sklearn MLP with scaling, robust defaults for small tabular data
        clf = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=16,
            learning_rate_init=1e-3,
            max_iter=400,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=25,
            random_state=seed,
        )
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    elif model_name == "hgb":
        # Histogram-based gradient boosting
        clf = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=3,
            max_iter=400,
            l2_regularization=0.0,
            early_stopping=True,
            random_state=seed,
        )
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    else:
        raise ValueError(f"Unknown model: {model_name}")


def fit_with_balance(model, X_train, y_train, balance_mode: str, seed: int):
    """
    balance_mode: none | weights | oversample
    Returns fitted model and the possibly resampled X_train,y_train, and sample_weight (or None).
    """
    rng = np.random.RandomState(seed)

    if balance_mode == "none":
        model.fit(X_train, y_train)
        return model

    if balance_mode == "weights":
        # inverse frequency weights
        n0 = np.sum(y_train == 0)
        n1 = np.sum(y_train == 1)
        # avoid div by zero
        w0 = 1.0 if n0 == 0 else (len(y_train) / (2.0 * n0))
        w1 = 1.0 if n1 == 0 else (len(y_train) / (2.0 * n1))
        sample_weight = np.where(y_train == 1, w1, w0).astype(float)
        # Pipeline passes sample_weight to last estimator if supported
        model.fit(X_train, y_train, clf__sample_weight=sample_weight)
        return model

    if balance_mode == "oversample":
        idx0 = np.where(y_train == 0)[0]
        idx1 = np.where(y_train == 1)[0]
        if len(idx0) == 0 or len(idx1) == 0:
            model.fit(X_train, y_train)
            return model
        if len(idx0) == len(idx1):
            model.fit(X_train, y_train)
            return model

        # duplicate minority with replacement to match majority
        if len(idx0) > len(idx1):
            maj, mino = idx0, idx1
        else:
            maj, mino = idx1, idx0

        extra = rng.choice(mino, size=(len(maj) - len(mino)), replace=True)
        new_idx = np.concatenate([maj, mino, extra])
        rng.shuffle(new_idx)
        Xb = X_train[new_idx]
        yb = y_train[new_idx]
        model.fit(Xb, yb)
        return model

    raise ValueError(f"Unknown balance mode: {balance_mode}")


# -----------------------------
# Main experiment loop
# -----------------------------

@dataclass
class FoldMetrics:
    bal_acc: float
    recall_pos: float
    spec_neg: float
    auroc: float
    auprc: float
    thr: float


def eval_one_fold(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    seed: int,
    thr_mode: str,
    thr_source: str,
    target_spec: float,
    balance_mode: str,
) -> FoldMetrics:
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_va, y_va = X[val_idx], y[val_idx]

    model = make_model(model_name, seed)

    if thr_source == "val":
        # Train on full train fold
        model = fit_with_balance(model, X_tr, y_tr, balance_mode, seed)
        p_val = model.predict_proba(X_va)[:, 1]
        # Choose threshold using validation fold (matches earlier scripts; slightly optimistic)
        if thr_mode == "youden":
            thr = pick_threshold_youden(y_va, p_val)
        elif thr_mode == "fixed_spec":
            thr = pick_threshold_fixed_specificity(y_va, p_val, target_spec=target_spec)
        elif thr_mode == "none":
            thr = 0.5
        else:
            raise ValueError("thr_mode must be youden, fixed_spec, or none")

    elif thr_source == "calib":
        # Split train fold into subtrain/calibration for threshold choice (cleaner)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
        (subtr_idx, calib_idx) = next(sss.split(X_tr, y_tr))
        X_sub, y_sub = X_tr[subtr_idx], y_tr[subtr_idx]
        X_cal, y_cal = X_tr[calib_idx], y_tr[calib_idx]

        model = fit_with_balance(model, X_sub, y_sub, balance_mode, seed)
        p_cal = model.predict_proba(X_cal)[:, 1]
        if thr_mode == "youden":
            thr = pick_threshold_youden(y_cal, p_cal)
        elif thr_mode == "fixed_spec":
            thr = pick_threshold_fixed_specificity(y_cal, p_cal, target_spec=target_spec)
        elif thr_mode == "none":
            thr = 0.5
        else:
            raise ValueError("thr_mode must be youden, fixed_spec, or none")

        p_val = model.predict_proba(X_va)[:, 1]

    else:
        raise ValueError("thr_source must be val or calib")

    y_hat = (p_val >= thr).astype(int)

    bal = balanced_accuracy_score(y_va, y_hat)
    rec = recall_score(y_va, y_hat, pos_label=1, zero_division=0)
    cm = confusion_matrix(y_va, y_hat, labels=[0, 1])
    spec = specificity_from_cm(cm)

    # Ranking metrics
    # Guard against single-class folds
    try:
        auroc = roc_auc_score(y_va, p_val)
    except Exception:
        auroc = float("nan")
    try:
        auprc = average_precision_score(y_va, p_val)
    except Exception:
        auprc = float("nan")

    return FoldMetrics(
        bal_acc=float(bal),
        recall_pos=float(rec),
        spec_neg=float(spec),
        auroc=float(auroc),
        auprc=float(auprc),
        thr=float(thr),
    )


def run_learning_curve(
    X: np.ndarray,
    y: np.ndarray,
    sizes: List[int],
    repeats: int,
    folds: int,
    model_name: str,
    thr_mode: str,
    thr_source: str,
    target_spec: float,
    balance_mode: str,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      per_run_df: metrics for each (size, repeat)
      summary_df: mean/std per size
    """
    N = len(y)
    rows = []
    rng = np.random.RandomState(seed)

    for n in sizes:
        n_eff = min(int(n), N)
        for r in range(repeats):
            # Stratified subsample of size n_eff
            if n_eff >= N:
                sub_idx = np.arange(N, dtype=int)
            else:
                sss = StratifiedShuffleSplit(
                    n_splits=1,
                    train_size=n_eff,
                    random_state=seed + 1000 * r + n_eff
                )
                sub_idx, _ = next(sss.split(np.zeros(N), y))
                sub_idx = np.array(sub_idx, dtype=int)


            Xs = X[sub_idx]
            ys = y[sub_idx]

            skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed + 10 * r + n_eff)

            fold_metrics = []
            for k, (tr, va) in enumerate(skf.split(Xs, ys)):
                fm = eval_one_fold(
                    model_name=model_name,
                    X=Xs,
                    y=ys,
                    train_idx=tr,
                    val_idx=va,
                    seed=seed + 100 * r + 7 * k + n_eff,
                    thr_mode=thr_mode,
                    thr_source=thr_source,
                    target_spec=target_spec,
                    balance_mode=balance_mode,
                )
                fold_metrics.append(fm)

            # average across folds for this repeat
            bal = np.mean([f.bal_acc for f in fold_metrics])
            rec = np.mean([f.recall_pos for f in fold_metrics])
            spec = np.mean([f.spec_neg for f in fold_metrics])
            auroc = np.nanmean([f.auroc for f in fold_metrics])
            auprc = np.nanmean([f.auprc for f in fold_metrics])
            thr = np.mean([f.thr for f in fold_metrics])

            rows.append(
                dict(
                    n=n_eff,
                    repeat=r + 1,
                    bal_acc=bal,
                    recall_abnormal=rec,
                    specificity_normal=spec,
                    auroc=auroc,
                    auprc=auprc,
                    thr=thr,
                )
            )

            print(
                f"[n={n_eff:3d} | rep {r+1:03d}] "
                f"bal={bal:.3f} rec={rec:.3f} spec={spec:.3f} auroc={auroc:.3f} auprc={auprc:.3f} thr={thr:.3f}"
            )

    per_run_df = pd.DataFrame(rows)

    def agg(col: str) -> pd.DataFrame:
        g = per_run_df.groupby("n")[col]
        return pd.DataFrame({f"{col}_mean": g.mean(), f"{col}_std": g.std(ddof=0)})

    summary = pd.concat(
        [
            agg("bal_acc"),
            agg("recall_abnormal"),
            agg("specificity_normal"),
            agg("auroc"),
            agg("auprc"),
            agg("thr"),
        ],
        axis=1,
    ).reset_index()

    return per_run_df, summary


def plot_metric(summary_df: pd.DataFrame, metric: str, out_png: str, fit_params: Optional[Dict[str, float]] = None):
    n = summary_df["n"].values
    m = summary_df[f"{metric}_mean"].values
    s = summary_df[f"{metric}_std"].values

    plt.figure()
    plt.errorbar(n, m, yerr=s, fmt="o", capsize=3)
    plt.xlabel("Number of subjects (n)")
    plt.ylabel(metric)
    plt.title(f"Learning curve: {metric}")

    if fit_params is not None:
        a, b, alpha = fit_params["a"], fit_params["b"], fit_params["alpha"]
        nn = np.linspace(np.min(n), np.max(n) * 2.0, 200)
        yy = a - b * (nn ** (-alpha))
        plt.plot(nn, yy)
        plt.xlim(np.min(n) * 0.95, np.max(n) * 2.0)

    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def parse_sizes(s: str, N: int) -> List[int]:
    if s.strip().lower() == "auto":
        # sensible defaults
        base = [25, 35, 45, 60, 75, 90, N]
        return sorted(list(dict.fromkeys([min(x, N) for x in base])))
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out = []
    for p in parts:
        out.append(int(p))
    out = [min(x, N) for x in out]
    out = sorted(list(dict.fromkeys(out)))
    return out


def parse_targets(s: str) -> Dict[str, float]:
    """
    Format: "auprc:0.85,bal_acc:0.65"
    """
    s = s.strip()
    if not s:
        return {}
    out = {}
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        k, v = item.split(":")
        out[k.strip()] = float(v.strip())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to combined_features.csv")
    ap.add_argument("--model", default="mlp", choices=["mlp", "hgb"])
    ap.add_argument("--sizes", default="auto", help="Comma list of n values, or 'auto'")
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--folds", type=int, default=5)

    ap.add_argument("--thr_mode", default="youden", choices=["youden", "fixed_spec", "none"])
    ap.add_argument("--thr_source", default="val", choices=["val", "calib"], help="val matches earlier scripts, calib is cleaner")
    ap.add_argument("--target_spec", type=float, default=0.90)

    ap.add_argument("--balance", default="none", choices=["none", "weights", "oversample"])
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--targets", default="auprc:0.85,bal_acc:0.65", help="Targets for projections, e.g. 'auprc:0.85,bal_acc:0.65'")
    ap.add_argument("--out_prefix", default="lc_bin")

    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    y_col = infer_label_column(df)

    y_multi = df[y_col].values
    y = to_binary_labels(y_multi)

    # drop label + common id-like cols
    drop_cols = [y_col, "CID", "cid", "PatientID", "patient_id", "id", "ID"]
    Xdf = numeric_feature_frame(df, drop_cols=drop_cols)
    X = Xdf.values.astype(np.float32)

    N = len(y)
    sizes = parse_sizes(args.sizes, N)

    print(f"Loaded: {args.csv}")
    print(f"X: {X.shape} y_bin: {y.shape}")
    uniq, cnt = np.unique(y, return_counts=True)
    print("Binary class counts:", dict(zip(uniq.tolist(), cnt.tolist())), "(0=Normal, 1=Altered)")
    print(f"Model={args.model} | repeats={args.repeats} | folds={args.folds} | thr_mode={args.thr_mode} | thr_source={args.thr_source} | balance={args.balance}")

    per_run_df, summary_df = run_learning_curve(
        X=X,
        y=y,
        sizes=sizes,
        repeats=args.repeats,
        folds=args.folds,
        model_name=args.model,
        thr_mode=args.thr_mode,
        thr_source=args.thr_source,
        target_spec=args.target_spec,
        balance_mode=args.balance,
        seed=args.seed,
    )

    out_csv_runs = f"{args.out_prefix}_per_run.csv"
    out_csv_sum = f"{args.out_prefix}_summary.csv"
    per_run_df.to_csv(out_csv_runs, index=False)
    summary_df.to_csv(out_csv_sum, index=False)

    # Fit curves and projections
    targets = parse_targets(args.targets)
    projections = {"model": args.model, "thr_mode": args.thr_mode, "thr_source": args.thr_source, "balance": args.balance, "targets": targets, "fits": {}, "required_n": {}}

    for metric in ["bal_acc", "auprc", "auroc", "recall_abnormal", "specificity_normal"]:
        y_mean = summary_df[f"{metric}_mean"].values
        n_vals = summary_df["n"].values.astype(float)

        # Only fit if enough points
        fit = fit_power_law(n_vals, y_mean)
        projections["fits"][metric] = fit

        if metric in targets:
            need = estimate_n_for_target(fit, targets[metric])
            projections["required_n"][metric] = need

        # Plot
        out_png = f"{args.out_prefix}_{metric}.png"
        plot_metric(summary_df, metric, out_png, fit_params=fit)

    out_json = f"{args.out_prefix}_projections.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(projections, f, indent=2)

    print("\nSaved:")
    print(" ", os.path.abspath(out_csv_runs))
    print(" ", os.path.abspath(out_csv_sum))
    print(" ", os.path.abspath(out_json))
    print(f" PNGs: {args.out_prefix}_<metric>.png")

    print("\nProjection targets:")
    for k, v in targets.items():
        need = projections["required_n"].get(k, None)
        if need is None:
            print(f"  {k} target={v} -> not reachable by fitted asymptote (or fit unstable)")
        else:
            print(f"  {k} target={v} -> estimated n ~ {need:.1f}")

if __name__ == "__main__":
    main()
