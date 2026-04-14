import argparse
import glob
import os
from datetime import datetime

import numpy as np
import pandas as pd


def is_number_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)


def bootstrap_ci(values, n_boot=2000, alpha=0.05, seed=42):
    """Bootstrap CI for the mean. Returns (mean, lo, hi)."""
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = []
    n = len(x)
    for _ in range(n_boot):
        samp = rng.choice(x, size=n, replace=True)
        means.append(np.mean(samp))
    means = np.asarray(means)
    lo = np.quantile(means, alpha / 2.0)
    hi = np.quantile(means, 1 - alpha / 2.0)
    return float(np.mean(x)), float(lo), float(hi)


def summarize_split_df(df: pd.DataFrame, exp_name: str, n_boot=2000, seed=42):
    """
    Produce one summary row per metric column (wide row).
    Keeps mean, std, median, p25, p75, and bootstrap CI for mean.
    """
    summary = {"experiment": exp_name, "n_splits": int(df["split_id"].nunique()) if "split_id" in df.columns else len(df)}

    metric_cols = []
    for c in df.columns:
        if c in ("split_id", "model", "thr_mode"):
            continue
        if is_number_series(df[c]):
            metric_cols.append(c)

    for c in metric_cols:
        vals = df[c].to_numpy(dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            continue
        summary[f"{c}__mean"] = float(np.mean(vals))
        summary[f"{c}__std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        summary[f"{c}__median"] = float(np.median(vals))
        summary[f"{c}__p25"] = float(np.quantile(vals, 0.25))
        summary[f"{c}__p75"] = float(np.quantile(vals, 0.75))
        m, lo, hi = bootstrap_ci(vals, n_boot=n_boot, seed=seed)
        summary[f"{c}__ci_lo"] = lo
        summary[f"{c}__ci_hi"] = hi

    return summary


def pretty_metric(mean, std):
    if np.isnan(mean):
        return ""
    return f"{mean:.4f} ± {std:.4f}"


def build_summary_table(summary_rows: list):
    df = pd.DataFrame(summary_rows)
    # Create "pretty" columns for common metrics if present
    pretty_cols = []
    for base in ["macro_f1", "bal_acc", "acc", "auroc", "auprc", "recall_abnormal", "specificity_normal", "thr"]:
        m = f"{base}__mean"
        s = f"{base}__std"
        if m in df.columns and s in df.columns:
            df[f"{base}__pretty"] = [pretty_metric(a, b) for a, b in zip(df[m].to_numpy(), df[s].to_numpy())]
            pretty_cols.append(f"{base}__pretty")
    # Put key columns in front
    front = ["experiment", "n_splits"] + pretty_cols
    rest = [c for c in df.columns if c not in front]
    return df[front + rest]


def try_paired_comparison(df_a, df_b, name_a, name_b, metrics, n_boot=2000, seed=42):
    """
    Pair by split_id and compute mean(diff) + bootstrap CI.
    Returns list of rows.
    """
    if "split_id" not in df_a.columns or "split_id" not in df_b.columns:
        return []

    merged = pd.merge(df_a, df_b, on="split_id", suffixes=(f"__{name_a}", f"__{name_b}"))
    rows = []
    if len(merged) == 0:
        return rows

    for m in metrics:
        ca = f"{m}__{name_a}"
        cb = f"{m}__{name_b}"
        if ca in merged.columns and cb in merged.columns and is_number_series(merged[ca]) and is_number_series(merged[cb]):
            diff = (merged[ca].to_numpy(dtype=float) - merged[cb].to_numpy(dtype=float))
            diff = diff[~np.isnan(diff)]
            if len(diff) == 0:
                continue
            mean_diff, lo, hi = bootstrap_ci(diff, n_boot=n_boot, seed=seed)
            rows.append({
                "A": name_a,
                "B": name_b,
                "metric": m,
                "mean(A-B)": mean_diff,
                "ci_lo": lo,
                "ci_hi": hi,
                "n": int(len(diff))
            })
    return rows


def load_csv_safe(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] Could not read {path}: {e}")
        return None


def summarize_oof_per_sample(oof_df: pd.DataFrame, exp_name: str):
    """
    Aggregate OOF rows per sample index if available.
    Supports:
    - multiclass: columns p0,p1,p2
    - binary: p_abnormal
    """
    idx_col = None
    for candidate in ["idx", "index", "sample_id", "ID"]:
        if candidate in oof_df.columns:
            idx_col = candidate
            break
    if idx_col is None:
        return None

    prob_cols = []
    for c in ["p0", "p1", "p2", "p_abnormal"]:
        if c in oof_df.columns and is_number_series(oof_df[c]):
            prob_cols.append(c)

    if not prob_cols:
        return None

    agg_dict = {}
    for c in prob_cols:
        agg_dict[f"{c}_mean"] = (c, "mean")
        agg_dict[f"{c}_std"] = (c, "std")
        agg_dict[f"{c}_min"] = (c, "min")
        agg_dict[f"{c}_max"] = (c, "max")

    # keep y_true-ish if exists
    ycol = None
    for candidate in ["y_true", "y_true_bin"]:
        if candidate in oof_df.columns:
            ycol = candidate
            break

    group_cols = [idx_col]
    if ycol is not None:
        group_cols.append(ycol)

    grouped = oof_df.groupby(group_cols).agg(**agg_dict).reset_index()
    grouped["experiment"] = exp_name

    # Sort: for binary, highest p_abnormal_mean first; for multiclass, highest p2_mean first if present
    if "p_abnormal_mean" in grouped.columns:
        grouped = grouped.sort_values("p_abnormal_mean", ascending=False)
    elif "p2_mean" in grouped.columns:
        grouped = grouped.sort_values("p2_mean", ascending=False)

    return grouped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default=".", help="Directorio con los CSVs")
    ap.add_argument("--out_xlsx", type=str, default="results_summary.xlsx")
    ap.add_argument("--out_csv", type=str, default="results_summary.csv")
    ap.add_argument("--n_boot", type=int, default=2000, help="Bootstrap resamples para CI")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    base_dir = os.path.abspath(args.dir)

    split_paths = sorted(glob.glob(os.path.join(base_dir, "*_split_metrics.csv")))
    oof_paths = sorted(glob.glob(os.path.join(base_dir, "*_oof.csv")))

    if not split_paths:
        raise SystemExit(f"No encuentro *_split_metrics.csv en {base_dir}")

    # Load all split dfs
    split_dfs = {}
    summary_rows = []
    thresholds_rows = []

    for p in split_paths:
        name = os.path.splitext(os.path.basename(p))[0].replace("_split_metrics", "")
        df = load_csv_safe(p)
        if df is None or len(df) == 0:
            continue

        # ensure split_id exists; if not, create one
        if "split_id" not in df.columns:
            df = df.copy()
            df["split_id"] = np.arange(1, len(df) + 1)

        split_dfs[name] = df

        summary_rows.append(summarize_split_df(df, name, n_boot=args.n_boot, seed=args.seed))

        # thresholds distribution if present
        if "thr" in df.columns and is_number_series(df["thr"]):
            for sid, thr in zip(df["split_id"].to_numpy(), df["thr"].to_numpy()):
                thresholds_rows.append({"experiment": name, "split_id": int(sid), "thr": float(thr)})

    summary_df = build_summary_table(summary_rows)
    summary_df.to_csv(os.path.join(base_dir, args.out_csv), index=False)

    # Paired comparisons: automatically try some likely pairs
    paired_rows = []

    # Heurística: si hay bin_R20_mlp_youden y bin_R20_hgb_youden, compararlos
    candidates = list(split_dfs.keys())
    pairs_to_try = []

    def add_pair_if_exists(a, b):
        if a in split_dfs and b in split_dfs:
            pairs_to_try.append((a, b))

    add_pair_if_exists("bin_R20_mlp_youden", "bin_R20_hgb_youden")
    add_pair_if_exists("bin_R20_hgb_youden", "bin_R20_hgb_fixed_spec")
    add_pair_if_exists("bin_R20_mlp_youden", "bin_R20_hgb_fixed_spec")

    # Also try exp1 vs exp3 if present
    add_pair_if_exists("exp1_mlp_R20", "exp3_boost_R20_sk_hgb")

    # metrics to compare if present
    common_metrics = [
        "macro_f1", "bal_acc", "acc",
        "auroc", "auprc",
        "recall_abnormal", "specificity_normal",
        "thr"
    ]

    for a, b in pairs_to_try:
        paired_rows += try_paired_comparison(
            split_dfs[a], split_dfs[b],
            name_a=a, name_b=b,
            metrics=common_metrics,
            n_boot=args.n_boot, seed=args.seed
        )

    paired_df = pd.DataFrame(paired_rows) if paired_rows else pd.DataFrame(columns=["A", "B", "metric", "mean(A-B)", "ci_lo", "ci_hi", "n"])

    thresholds_df = pd.DataFrame(thresholds_rows) if thresholds_rows else pd.DataFrame(columns=["experiment", "split_id", "thr"])

    # OOF per-sample summaries
    oof_sample_tables = []
    for p in oof_paths:
        name = os.path.splitext(os.path.basename(p))[0].replace("_oof", "")
        oof = load_csv_safe(p)
        if oof is None or len(oof) == 0:
            continue
        agg = summarize_oof_per_sample(oof, name)
        if agg is not None:
            oof_sample_tables.append(agg)

    oof_samples_df = pd.concat(oof_sample_tables, ignore_index=True) if oof_sample_tables else pd.DataFrame()

    # Write Excel
    out_xlsx_path = os.path.join(base_dir, args.out_xlsx)
    with pd.ExcelWriter(out_xlsx_path, engine="openpyxl") as writer:
        # Header sheet
        info = pd.DataFrame({
            "generated_at": [datetime.now().isoformat(timespec="seconds")],
            "directory": [base_dir],
            "n_split_files": [len(split_paths)],
            "n_oof_files": [len(oof_paths)],
            "bootstrap_n": [args.n_boot],
            "seed": [args.seed],
        })
        info.to_excel(writer, sheet_name="Info", index=False)

        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        paired_df.to_excel(writer, sheet_name="PairedComparisons", index=False)

        if len(thresholds_df) > 0:
            thresholds_df.to_excel(writer, sheet_name="Thresholds", index=False)

        # One sheet per experiment split metrics (limit sheet name to 31 chars)
        for name, df in split_dfs.items():
            sheet = name[:31]
            df.to_excel(writer, sheet_name=sheet, index=False)

        if len(oof_samples_df) > 0:
            # Could be large; still useful
            oof_samples_df.to_excel(writer, sheet_name="OOF_per_sample", index=False)

    print("\nDONE")
    print(f"Saved:\n  {out_xlsx_path}\n  {os.path.join(base_dir, args.out_csv)}")

    # Quick console “headline”
    print("\n=== HEADLINE (from Summary) ===")
    head_cols = [c for c in summary_df.columns if c.endswith("__pretty")]
    show_cols = ["experiment", "n_splits"] + head_cols
    show_cols = [c for c in show_cols if c in summary_df.columns]
    print(summary_df[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
