#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, accuracy_score,
    confusion_matrix, classification_report
)

# -------------------------
# Logging
# -------------------------
def setup_logger(out_dir: Path, level: str = "INFO") -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("option4_v2")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(out_dir / "pipeline.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# -------------------------
# Helpers
# -------------------------
def compute_weights(X_abs: pd.DataFrame, mode: str, eps: float = 1e-9):
    """
    X_abs: gramos/año por CID (N x C)
    mode:
      - "norm"    : grams / sum(grams)
      - "lognorm" : log1p(grams) / sum(log1p(grams))
      - "abs"     : grams (sin normalizar)
      - "log"     : log1p(grams) (sin normalizar)
    """
    grams = X_abs.values.astype(float)
    total_grams = grams.sum(axis=1)

    if mode == "norm":
        denom = np.maximum(total_grams, eps)
        w = grams / denom[:, None]
        w_name = "grams/sum(grams)"
    elif mode == "lognorm":
        lg = np.log1p(np.maximum(grams, 0.0))
        denom = np.maximum(lg.sum(axis=1), eps)
        w = lg / denom[:, None]
        w_name = "log1p(grams)/sum(log1p(grams))"
    elif mode == "abs":
        w = grams
        w_name = "grams (absolute)"
    elif mode == "log":
        w = np.log1p(np.maximum(grams, 0.0))
        w_name = "log1p(grams) (absolute)"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return w, total_grams, w_name


def get_models(seed: int):
    return {
        "LogReg_multinomial": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=5000,
                multi_class="multinomial",
                class_weight="balanced"
            ))
        ]),
        "LinearSVC": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LinearSVC(class_weight="balanced"))
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=800,
            random_state=seed,
            class_weight="balanced_subsample",
            n_jobs=-1
        )
    }


def extract_importance(model, feat_names):
    """
    Importancias:
    - LogReg/LinearSVC: media abs(coef) entre clases
    - RF: feature_importances_
    """
    if isinstance(model, Pipeline):
        clf = model.named_steps.get("clf", None)
        if clf is None:
            return None
        if hasattr(clf, "coef_"):
            coefs = np.array(clf.coef_)
            return np.mean(np.abs(coefs), axis=0)
        return None

    if hasattr(model, "feature_importances_"):
        return np.array(model.feature_importances_)
    return None


def run_cv(X, y, feat_names, models, n_splits, seed, out_dir, variant_name, logger):
    labels = np.unique(y)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    variant_dir = out_dir / f"results_{variant_name}"
    variant_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    logger.info(f"=== CV variant '{variant_name}' | X={X.shape} | y-dist={pd.Series(y).value_counts().to_dict()} ===")

    for mname, model in models.items():
        logger.info(f"[{variant_name}] Model: {mname} | starting CV...")

        y_pred = np.empty_like(y, dtype=object)
        fold_imps = []
        fold_metrics = []

        for fold, (tr, te) in enumerate(cv.split(X, y), start=1):
            model.fit(X[tr], y[tr])
            pred = model.predict(X[te])
            y_pred[te] = pred

            fm = {
                "fold": fold,
                "f1_macro": float(f1_score(y[te], pred, average="macro")),
                "balanced_acc": float(balanced_accuracy_score(y[te], pred)),
                "acc": float(accuracy_score(y[te], pred)),
            }
            fold_metrics.append(fm)

            imp = extract_importance(model, feat_names)
            if imp is not None:
                fold_imps.append(imp)

            logger.info(f"  fold {fold}/{n_splits} | f1_macro={fm['f1_macro']:.4f} "
                        f"| bal_acc={fm['balanced_acc']:.4f} | acc={fm['acc']:.4f}")

        oof_f1 = float(f1_score(y, y_pred, average="macro"))
        oof_bal = float(balanced_accuracy_score(y, y_pred))
        oof_acc = float(accuracy_score(y, y_pred))

        cm = confusion_matrix(y, y_pred, labels=labels)
        cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])

        cm_df.to_csv(variant_dir / f"{mname}_confusion_matrix.csv")
        with open(variant_dir / f"{mname}_classification_report.txt", "w", encoding="utf-8") as f:
            f.write(classification_report(y, y_pred, digits=4))
        pd.DataFrame({"y_true": y, "y_pred": y_pred}).to_csv(variant_dir / f"{mname}_oof_predictions.csv", index=False)

        logger.info(f"[{variant_name}] {mname} OOF: f1_macro={oof_f1:.4f} | bal_acc={oof_bal:.4f} | acc={oof_acc:.4f}")

        top_feat_file = ""
        if len(fold_imps) > 0:
            mean_imp = np.mean(np.stack(fold_imps, axis=0), axis=0)
            imp_df = pd.DataFrame({"feature": feat_names, "importance": mean_imp}).sort_values("importance", ascending=False)
            top_feat_file = f"{mname}_top_features.csv"
            imp_df.to_csv(variant_dir / top_feat_file, index=False)
            logger.info("  top 10: " + ", ".join(imp_df["feature"].head(10).tolist()))

        fm_df = pd.DataFrame(fold_metrics)
        rows.append({
            "variant": variant_name,
            "model": mname,
            "n_samples": int(len(y)),
            "n_features": int(X.shape[1]),
            "cv_folds": int(n_splits),
            "oof_f1_macro": oof_f1,
            "oof_balanced_acc": oof_bal,
            "oof_acc": oof_acc,
            "fold_f1_macro_mean": float(fm_df["f1_macro"].mean()),
            "fold_f1_macro_std": float(fm_df["f1_macro"].std(ddof=0)),
            "top_features_file": top_feat_file
        })

    return rows


# -------------------------
# Main
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Opción 4 (A/B/C/D): Dieta vs MolAgg vs ProtAgg vs Combinado, con log completo."
    )
    ap.add_argument("--base-dieta", required=True, help="CSV: 1ª col ID, última col target, resto CIDs (gramos/año).")
    ap.add_argument("--molecular-features", required=True, help="CSV: CID, Proteina, docking, descriptores numéricos...")
    ap.add_argument("--out-dir", required=True, help="Directorio de salida.")

    ap.add_argument("--id-col", default=None, help="Nombre columna ID (default: primera).")
    ap.add_argument("--target-col", default=None, help="Nombre columna target (default: última).")

    ap.add_argument("--weighting", default="norm", choices=["norm", "lognorm", "abs", "log"],
                    help="Cómo transformar gramos->pesos para A y MolAgg (default norm).")
    ap.add_argument("--prot-agg-mode", default="dose", choices=["dose", "comp"],
                    help="ProtAgg: 'dose'=gramos×potencia, 'comp'=pesos×potencia (default dose).")

    ap.add_argument("--dock-col", default="Affinity1_rigid", choices=["Affinity1_rigid", "Affinity2_flex"],
                    help="Columna de docking a usar como potencia principal.")

    ap.add_argument("--protagg_numeric_mode", default="dock_only", choices=["dock_only", "all_numeric_non_smiles"],
                    help="ProtAgg usa solo docking o docking+otros numéricos no-SMILES (default dock_only).")

    ap.add_argument("--cv", type=int, default=5, help="Folds CV (default 5; se ajusta a clase minoritaria).")
    ap.add_argument("--seed", type=int, default=42, help="Seed.")
    ap.add_argument("--log-level", default="INFO", help="INFO/DEBUG.")
    ap.add_argument("--save-datasets", action="store_true", help="Guarda datasets A/B/C/D.")
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    logger = setup_logger(out_dir, args.log_level)

    logger.info("========== OPTION 4 ABLATION PIPELINE (v2) ==========")
    logger.info("Args:\n" + json.dumps(vars(args), indent=2, ensure_ascii=False))

    # Load base_dieta
    base = pd.read_csv(args.base_dieta)
    id_col = args.id_col if args.id_col else base.columns[0]
    y_col = args.target_col if args.target_col else base.columns[-1]
    cid_cols = [c for c in base.columns if c not in [id_col, y_col]]

    logger.info(f"Loaded base_dieta: shape={base.shape}")
    logger.info(f"ID col='{id_col}' | target col='{y_col}' | #CID cols={len(cid_cols)}")
    logger.info("Target distribution: " + str(base[y_col].value_counts().to_dict()))

    # X_abs (gramos por CID)
    X_abs_df = base[cid_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    y = base[y_col].astype(str).values
    ids = base[id_col].astype(str).values

    # Pesos para A y MolAgg
    W, total_grams, w_name = compute_weights(X_abs_df, args.weighting)
    logger.info(f"Weighting='{args.weighting}' => {w_name}")
    logger.info(f"Total grams stats: min={total_grams.min():.3f}, p50={np.median(total_grams):.3f}, max={total_grams.max():.3f}")

    # Load molecular_features
    mf = pd.read_csv(args.molecular_features)
    logger.info(f"Loaded molecular_features: shape={mf.shape}")

    # Columnas mínimas
    needed = ["CID", "Proteina", "Affinity1_rigid", "Affinity2_flex"]
    for c in needed:
        if c not in mf.columns:
            raise ValueError(f"molecular_features missing required column: {c}")

    # Normalizar tipos
    mf = mf.copy()
    mf["CID"] = mf["CID"].astype(str).str.strip()
    mf["Proteina"] = mf["Proteina"].astype(str).str.strip()

    base_cids = [str(c).strip() for c in cid_cols]
    # Preserve the receptor feature-block order used by the classifiers.
    original_feature_order = ["CYP1A1_4I8V", "CYP1A2_2HI4", "CYP3A4_1TQN", "CYP2E1_3KOH"]
    proteins = [p for p in original_feature_order if p in mf["Proteina"].unique()]
    proteins.extend(p for p in sorted(mf["Proteina"].unique()) if p not in proteins)

    logger.info(f"Proteins found ({len(proteins)}): {proteins}")
    if len(proteins) != 4:
        logger.warning("OJO: esperabas 4 proteínas. El fichero tiene " + str(len(proteins)))

    # --- Selección de columnas numéricas ---
    # Nunca usar textos: smiles, nombre, Tipo, etc.
    # Nunca usar CID como feature.
    numeric_cols = [c for c in mf.columns if pd.api.types.is_numeric_dtype(mf[c])]
    # numeric_cols incluye docking + SMILES_* + canales c0_... etc. Ojo: CID es str ya, así que no entra.
    logger.info(f"Numeric columns in molecular_features: {numeric_cols}")

    # MolAgg: SOLO descriptores globales SMILES_* numéricos (no textos)
    mol_cols = [c for c in numeric_cols if c.startswith("SMILES_")]
    logger.info(f"MolAgg numeric SMILES_* columns ({len(mol_cols)}): {mol_cols}")

    # ProtAgg: por proteína. Dos modos:
    if args.protagg_numeric_mode == "dock_only":
        prot_cols = [args.dock_col]
    else:
        # docking + otros numéricos NO SMILES (p.ej. canales de interacción)
        prot_cols = [c for c in numeric_cols if (not c.startswith("SMILES_"))]
        if args.dock_col not in prot_cols:
            prot_cols = [args.dock_col] + prot_cols

    logger.info(f"ProtAgg columns mode='{args.protagg_numeric_mode}' => ({len(prot_cols)}): {prot_cols}")

    # ----- MolAgg matrix M_desc (C x F) -----
    if len(mol_cols) > 0:
        mf_desc = mf.groupby("CID")[mol_cols].mean(numeric_only=True)
        M_desc = mf_desc.reindex(base_cids).fillna(0.0)  # C x F
        missing = int((M_desc.sum(axis=1) == 0).sum())
        logger.info(f"MolAgg: CIDs base={len(base_cids)} | CIDs sin descriptores (0)={missing}")

        MolAgg = W @ M_desc.values  # N x F
        mol_feat_names = [f"MolAgg__{c}" for c in mol_cols]
    else:
        MolAgg = np.zeros((len(y), 0), dtype=float)
        mol_feat_names = []

    # ----- ProtAgg tables per protein (C x K) -----
    # Para la potencia de docking: convertimos energía negativa a "fuerza positiva"
    # potency = max(0, -Affinity)
    # (esto es simple y explicable)
    prot_blocks = []
    prot_feat_names = []

    # Pesos para ProtAgg: dose o comp
    if args.prot_agg_mode == "dose":
        Wp = X_abs_df.values  # gramos (dosis)
        logger.info("ProtAgg mode='dose': usando gramos como dosis.")
    else:
        Wp = W  # composición
        logger.info("ProtAgg mode='comp': usando pesos normalizados como composición.")

    for p in proteins:
        sub = mf[mf["Proteina"] == p].copy()

        tab = sub.set_index("CID")[prot_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        tab = tab.groupby(tab.index).mean()
        tab = tab.reindex(base_cids).fillna(0.0)

        # Transformación de docking -> potencia
        if args.dock_col in tab.columns:
            tab[args.dock_col] = (-tab[args.dock_col]).clip(lower=0.0)

        block = Wp @ tab.values  # N x K

        names = [f"ProtAgg__{p}__{c}" for c in tab.columns]
        prot_blocks.append(block)
        prot_feat_names.extend(names)

        logger.info(f"ProtAgg '{p}': built block N x {block.shape[1]} | "
                    f"CIDs sin fila para esta proteína (0)={(tab.sum(axis=1)==0).sum()}")

    ProtAgg = np.concatenate(prot_blocks, axis=1) if prot_blocks else np.zeros((len(y), 0), dtype=float)

    logger.info(f"Feature blocks built => Diet={W.shape[1]}+1 | MolAgg={MolAgg.shape[1]} | ProtAgg={ProtAgg.shape[1]}")

    # ----- Variants A/B/C/D -----
    # A) Dieta: pesos por CID + total_grams
    A_X = np.concatenate([W, total_grams[:, None]], axis=1)
    A_names = [f"DietW__{c}" for c in base_cids] + ["Diet__total_grams"]

    # B) MolAgg + total_grams
    B_X = np.concatenate([MolAgg, total_grams[:, None]], axis=1) if MolAgg.shape[1] > 0 else total_grams[:, None]
    B_names = mol_feat_names + ["Diet__total_grams"]

    # C) ProtAgg + total_grams
    C_X = np.concatenate([ProtAgg, total_grams[:, None]], axis=1) if ProtAgg.shape[1] > 0 else total_grams[:, None]
    C_names = prot_feat_names + ["Diet__total_grams"]

    # D) Combined = Diet + MolAgg + ProtAgg
    D_X = np.concatenate([A_X, MolAgg, ProtAgg], axis=1) if (MolAgg.shape[1] + ProtAgg.shape[1] > 0) else A_X
    D_names = A_names + mol_feat_names + prot_feat_names

    # E) NoDock = Diet + MolAgg
    # Esta variante conserva dieta + descriptores moleculares simples,
    # pero elimina todas las features derivadas del docking/proteínas.
    E_X = np.concatenate([A_X, MolAgg], axis=1) if MolAgg.shape[1] > 0 else A_X
    E_names = A_names + mol_feat_names

    variants = {
        "A_diet": (A_X, A_names),
        "B_molagg": (B_X, B_names),
        "E_nodock": (E_X, E_names),
        "C_protagg": (C_X, C_names),
        "D_combined": (D_X, D_names)
    }

    if args.save_datasets:
        logger.info("Saving datasets A/B/C/D ...")
        for vname, (Xv, fnames) in variants.items():
            df_out = pd.DataFrame(Xv, columns=fnames)
            df_out.insert(0, "ID", ids)
            df_out.insert(1, "y", y)
            out_path = out_dir / f"dataset_{vname}.csv"
            df_out.to_csv(out_path, index=False)
            logger.info(f"  saved {out_path.name} shape={df_out.shape}")

    # CV folds safe
    counts = pd.Series(y).value_counts()
    min_count = int(counts.min())
    n_splits = min(args.cv, min_count)
    if n_splits < 2:
        raise ValueError(f"Clase minoritaria demasiado pequeña para CV: min_count={min_count}")
    if n_splits != args.cv:
        logger.warning(f"Adjusting CV folds {args.cv} -> {n_splits} (min class count={min_count})")

    models = get_models(args.seed)

    all_rows = []
    for vname, (Xv, fnames) in variants.items():
        all_rows.extend(run_cv(
            Xv, y, fnames,
            models=models,
            n_splits=n_splits,
            seed=args.seed,
            out_dir=out_dir,
            variant_name=vname,
            logger=logger
        ))

    summary = pd.DataFrame(all_rows).sort_values(["variant", "oof_f1_macro"], ascending=[True, False])
    summary.to_csv(out_dir / "metrics_summary.csv", index=False)

    with open(out_dir / "REPORT.txt", "w", encoding="utf-8") as f:
        f.write("OPTION 4 ABLATION REPORT (v2)\n")
        f.write("=============================\n\n")
        f.write(f"base_dieta: {args.base_dieta}\n")
        f.write(f"molecular_features: {args.molecular_features}\n\n")
        f.write(f"Weighting: {args.weighting} | ProtAgg mode: {args.prot_agg_mode}\n")
        f.write(f"Dock col: {args.dock_col} | ProtAgg numeric mode: {args.protagg_numeric_mode}\n\n")
        f.write(f"Proteins ({len(proteins)}): {proteins}\n")
        f.write(f"MolAgg cols ({len(mol_cols)}): {mol_cols}\n")
        f.write(f"ProtAgg cols ({len(prot_cols)}): {prot_cols}\n\n")
        f.write("Target distribution:\n")
        f.write(str(counts.to_dict()) + "\n\n")
        f.write("Metrics summary (sorted by oof_f1_macro overall):\n")
        f.write(summary.sort_values("oof_f1_macro", ascending=False).to_string(index=False))
        f.write("\n")

    logger.info("Saved: pipeline.log, REPORT.txt, metrics_summary.csv + results_* folders")
    logger.info("DONE ✅")

if __name__ == "__main__":
    main()
