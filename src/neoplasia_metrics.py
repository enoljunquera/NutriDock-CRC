# neoplasia_metrics.py
# Calcula métricas centradas en NEOPLASIA (clase 2) a partir de los *_oof.csv
# generados por exp1_repeated_mlp.py y exp3_boosting_repeated.py

import argparse
import numpy as np
import pandas as pd


NEO = 2  # etiqueta de Neoplasia


def safe_div(a, b):
    return float(a) / float(b) if b != 0 else float("nan")


def bin_metrics(y_true_bin, y_pred_bin):
    # y_*_bin son arrays 0/1 (1 = neoplasia)
    tp = int(np.sum((y_true_bin == 1) & (y_pred_bin == 1)))
    fp = int(np.sum((y_true_bin == 0) & (y_pred_bin == 1)))
    fn = int(np.sum((y_true_bin == 1) & (y_pred_bin == 0)))
    tn = int(np.sum((y_true_bin == 0) & (y_pred_bin == 0)))

    prec = safe_div(tp, tp + fp)
    rec = safe_div(tp, tp + fn)
    f1 = safe_div(2 * prec * rec, prec + rec) if np.isfinite(prec) and np.isfinite(rec) else float("nan")
    spec = safe_div(tn, tn + fp)

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "specificity": spec
    }


def threshold_sweep(p2, y_true, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)

    y_true_bin = (y_true == NEO).astype(int)

    rows = []
    for t in thresholds:
        y_pred_bin = (p2 >= t).astype(int)
        m = bin_metrics(y_true_bin, y_pred_bin)
        m.update({"threshold": float(t)})
        rows.append(m)

    df = pd.DataFrame(rows)
    # criterio por defecto: máximo F1, y si empata, mayor recall
    df_sorted = df.sort_values(["f1", "recall"], ascending=[False, False]).reset_index(drop=True)
    return df_sorted


def per_split_stats(df):
    """
    df: oof_df con columnas split_id, y_true, y_pred, p2
    Devuelve un DataFrame con métricas por split.
    """
    out = []
    for sid, g in df.groupby("split_id"):
        y_true = g["y_true"].to_numpy(dtype=int)
        y_pred = g["y_pred"].to_numpy(dtype=int)

        y_true_bin = (y_true == NEO).astype(int)
        y_pred_bin = (y_pred == NEO).astype(int)

        m = bin_metrics(y_true_bin, y_pred_bin)
        m.update({
            "split_id": int(sid),
            "n_total": int(len(g)),
            "n_true_neoplasia": int(np.sum(y_true_bin)),
            "n_pred_neoplasia": int(np.sum(y_pred_bin)),
            "hit_any_neoplasia": int(m["tp"] > 0)
        })
        out.append(m)

    return pd.DataFrame(out).sort_values("split_id")


def mean_std(series):
    return float(np.nanmean(series)), float(np.nanstd(series, ddof=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof_csv", type=str, required=True,
                    help="Ruta al *_oof.csv (exp1_mlp_R20_oof.csv, exp3_boost_R20_*_oof.csv, etc.)")
    ap.add_argument("--out_prefix", type=str, default="neo_metrics",
                    help="Prefijo de salida para CSVs")
    ap.add_argument("--idx_col", type=str, default="idx",
                    help="Nombre de columna del índice de muestra (por defecto 'idx')")
    args = ap.parse_args()

    oof = pd.read_csv(args.oof_csv)

    required = {"split_id", "y_true", "y_pred", "p2"}
    missing_cols = required - set(oof.columns)
    if missing_cols:
        raise ValueError(f"Faltan columnas en {args.oof_csv}: {missing_cols}\n"
                         f"Se esperan al menos: {sorted(required)}")

    # --- 1) Métricas por split (con el argmax ya hecho) ---
    split_df = per_split_stats(oof)

    # resumen global (distribución a lo largo de splits)
    prec_mean, prec_std = mean_std(split_df["precision"])
    rec_mean, rec_std = mean_std(split_df["recall"])
    f1_mean, f1_std = mean_std(split_df["f1"])
    spec_mean, spec_std = mean_std(split_df["specificity"])

    hit_rate = float(np.mean(split_df["hit_any_neoplasia"]))  # proporción de splits donde TP>0
    mean_true_neo = float(np.mean(split_df["n_true_neoplasia"]))
    mean_pred_neo = float(np.mean(split_df["n_pred_neoplasia"]))

    print("\n=== NEOPLASIA METRICS (per-split distribution, using y_pred argmax) ===")
    print(f"precision:   {prec_mean:.4f} ± {prec_std:.4f}")
    print(f"recall:      {rec_mean:.4f} ± {rec_std:.4f}")
    print(f"f1:          {f1_mean:.4f} ± {f1_std:.4f}")
    print(f"specificity: {spec_mean:.4f} ± {spec_std:.4f}")
    print(f"hit_rate (TP>0): {hit_rate:.3f}  (fraction of splits that detect at least 1 neoplasia)")
    print(f"mean true neoplasias per split: {mean_true_neo:.2f}")
    print(f"mean predicted neoplasias per split: {mean_pred_neo:.2f}")

    # Guardar por-split
    split_out = f"{args.out_prefix}_per_split.csv"
    split_df.to_csv(split_out, index=False)
    print(f"\nSaved per-split metrics -> {split_out}")

    # --- 2) Threshold sweep global (p2 vs not) ---
    # Esto NO es nested (lo calculas con todo el OOF junto). Úsalo como análisis exploratorio.
    y_true_all = oof["y_true"].to_numpy(dtype=int)
    p2_all = oof["p2"].to_numpy(dtype=float)

    sweep = threshold_sweep(p2_all, y_true_all)
    best = sweep.iloc[0].to_dict()

    print("\n=== THRESHOLD SWEEP (global OOF, exploratory) ===")
    print(f"Best threshold (by F1 then recall): t={best['threshold']:.2f}")
    print(f"  precision={best['precision']:.4f} recall={best['recall']:.4f} f1={best['f1']:.4f} spec={best['specificity']:.4f}")
    print(f"  TP={int(best['tp'])} FP={int(best['fp'])} FN={int(best['fn'])} TN={int(best['tn'])}")

    sweep_out = f"{args.out_prefix}_threshold_sweep.csv"
    sweep.to_csv(sweep_out, index=False)
    print(f"Saved threshold sweep -> {sweep_out}")

    # --- 3) Si quieres: “ensemble” por muestra (promedio de p2 sobre splits)
    # Útil para ver sujetos consistentemente marcados como neoplasia.
    if args.idx_col in oof.columns:
        agg = (
            oof.groupby(args.idx_col)
            .agg(y_true=("y_true", "first"),
                 p2_mean=("p2", "mean"),
                 p2_std=("p2", "std"),
                 p2_min=("p2", "min"),
                 p2_max=("p2", "max"),
                 n=("p2", "count"))
            .reset_index()
            .sort_values("p2_mean", ascending=False)
        )
        agg_out = f"{args.out_prefix}_per_sample_p2.csv"
        agg.to_csv(agg_out, index=False)
        print(f"\nSaved per-sample p2 summary -> {agg_out}")
        print("Top 10 samples by p2_mean:")
        print(agg.head(10).to_string(index=False))
    else:
        print(f"\nNota: no encuentro la columna '{args.idx_col}' en el OOF, así que no hago el resumen por muestra.")


if __name__ == "__main__":
    main()
