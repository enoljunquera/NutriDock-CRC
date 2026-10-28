import argparse
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def holm_adjust(pvals):
    """Holm step-down adjusted p-values, preserving original order."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj_sorted = np.empty(m, dtype=float)

    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adj_sorted[rank] = min(running, 1.0)

    out = np.empty(m, dtype=float)
    for rank, idx in enumerate(order):
        out[idx] = adj_sorted[rank]
    return out


def repetition_means(df, metrics, folds_per_repeat=5):
    if "split_id" not in df.columns:
        raise ValueError("El CSV no contiene split_id.")

    df = df.copy().sort_values("split_id")
    split_ids = df["split_id"].astype(int).to_numpy()

    expected = np.arange(1, len(df) + 1)
    if not np.array_equal(split_ids, expected):
        raise ValueError(
            "split_id debe ser consecutivo 1..N para reconstruir las repeticiones "
            "exactamente como RepeatedStratifiedKFold."
        )

    if len(df) % folds_per_repeat != 0:
        raise ValueError(
            f"El número de splits ({len(df)}) no es múltiplo de {folds_per_repeat}."
        )

    missing = [m for m in metrics if m not in df.columns]
    if missing:
        raise ValueError(f"Faltan métricas en el CSV: {missing}")

    df["repeat"] = ((df["split_id"] - 1) // folds_per_repeat) + 1
    rep = df.groupby("repeat", sort=True)[metrics].mean()

    counts = df.groupby("repeat").size()
    if not np.all(counts.to_numpy() == folds_per_repeat):
        raise ValueError("Alguna repetición no contiene exactamente 5 folds.")

    return rep


def paired_family(df_mlp, df_hgb, metrics, family, folds_per_repeat=5):
    a = repetition_means(df_mlp, metrics, folds_per_repeat)
    b = repetition_means(df_hgb, metrics, folds_per_repeat)

    if not a.index.equals(b.index):
        raise ValueError(f"Las repeticiones MLP/HGB no coinciden en {family}.")

    rows = []
    for metric in metrics:
        x = a[metric].to_numpy(dtype=float)
        y = b[metric].to_numpy(dtype=float)
        d = x - y

        if np.allclose(d, 0.0):
            stat, p = 0.0, 1.0
        else:
            res = wilcoxon(
                x,
                y,
                alternative="two-sided",
                zero_method="wilcox",
                correction=False,
                method="auto",
            )
            stat, p = float(res.statistic), float(res.pvalue)

        rows.append({
            "family": family,
            "metric": metric,
            "n_repeats": len(x),
            "mlp_mean_across_repeats": float(np.mean(x)),
            "hgb_mean_across_repeats": float(np.mean(y)),
            "mean_diff_mlp_minus_hgb": float(np.mean(d)),
            "median_diff_mlp_minus_hgb": float(np.median(d)),
            "wilcoxon_W": stat,
            "p_raw": p,
        })

    out = pd.DataFrame(rows)
    out["p_holm"] = holm_adjust(out["p_raw"].to_numpy())
    out["significant_holm_0.05"] = out["p_holm"] < 0.05
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Comparaciones pareadas MLP vs HGB a nivel de repetición: Wilcoxon + Holm."
    )
    ap.add_argument(
        "--bin_mlp",
        default="13_bin_mlp_nested_corregido_mlp_youden_split_metrics.csv",
    )
    ap.add_argument(
        "--bin_hgb",
        default="15_bin_hgb_corregido_hgb_youden_split_metrics.csv",
    )
    ap.add_argument(
        "--three_mlp",
        default="10_threeclass_mlp_corregido_split_metrics.csv",
    )
    ap.add_argument(
        "--three_hgb",
        default="11_threeclass_hgb_corregido_sk_hgb_split_metrics.csv",
    )
    ap.add_argument("--folds_per_repeat", type=int, default=5)
    ap.add_argument(
        "--out",
        default="21_paired_wilcoxon_holm_corregido.csv",
    )
    args = ap.parse_args()

    bin_mlp = pd.read_csv(args.bin_mlp)
    bin_hgb = pd.read_csv(args.bin_hgb)
    three_mlp = pd.read_csv(args.three_mlp)
    three_hgb = pd.read_csv(args.three_hgb)

    binary_metrics = [
        "bal_acc",
        "recall_abnormal",
        "specificity_normal",
        "acc",
        "f1",
        "auroc",
        "auprc",
    ]
    three_metrics = ["macro_f1", "bal_acc", "acc"]

    binary = paired_family(
        bin_mlp,
        bin_hgb,
        binary_metrics,
        family="binary_youden",
        folds_per_repeat=args.folds_per_repeat,
    )
    three = paired_family(
        three_mlp,
        three_hgb,
        three_metrics,
        family="three_class",
        folds_per_repeat=args.folds_per_repeat,
    )

    result = pd.concat([binary, three], ignore_index=True)
    result.to_csv(args.out, index=False)

    print("\n=== BINARY YOUDEN: MLP vs HGB ===")
    print(
        binary[
            ["metric", "mean_diff_mlp_minus_hgb", "p_raw", "p_holm", "significant_holm_0.05"]
        ].to_string(index=False)
    )

    print("\n=== THREE-CLASS: MLP vs HGB ===")
    print(
        three[
            ["metric", "mean_diff_mlp_minus_hgb", "p_raw", "p_holm", "significant_holm_0.05"]
        ].to_string(index=False)
    )

    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
