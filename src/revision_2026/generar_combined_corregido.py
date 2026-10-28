#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Construct exposure-weighted Diet, MolAgg, ProtAgg and Combined features.

Requires molecular descriptors and an authorized cohort-level dietary table.
The published molecular table is available under data/molecular_features.csv.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from normalizar_molecular_features import canonicalize

ROOT = Path(__file__).resolve().parent

MOL_PATH = next((p for p in [
    ROOT.parent.parent / "data" / "molecular_features.csv",
    ROOT / "07_dataset_molecular_corregido" / "molecular_features_CORREGIDO.csv",
] if p.is_file()), ROOT.parent.parent / "data" / "molecular_features.csv")

DIET_CANDIDATES = [
    ROOT / "base_dieta.csv",
    ROOT / "CYP1A1_4I8V" / "base_dieta.csv",
    ROOT / "CYP1A2_2HI4" / "base_dieta.csv",
    ROOT / "CYP2E1_3KOH" / "base_dieta.csv",
]

DESC_COLS = ["SMILES_MW", "SMILES_TPSA", "SMILES_logP", "SMILES_NumRotBonds"]

PROT_COLS = [
    "Affinity1_rigid","Affinity2_flex",
    "c0_core_p95","c1_core_p95",
    "c2_core_max","c2_core_p95",
    "b4_core_mean","b7_core_mean",
    "c2_shell1_max","c2_shell1_p95",
    "b4_shell1_mean","b7_shell1_mean",
    "c2_shell2_max","c2_shell2_p95",
    "b4_shell2_mean","b7_shell2_mean",
]

# Protein feature blocks are arranged in the order used for training.
PROTEINS = ["CYP1A1_4I8V","CYP1A2_2HI4","CYP3A4_1TQN","CYP2E1_3KOH"]

LABEL_MAP = {"Normal": 0, "Pólipo": 1, "Neoplasia": 2}


def build_combined_features(
    diet_df: pd.DataFrame,
    mol_df: pd.DataFrame,
    weighting: str = "norm",
    prot_mode: str = "full",
    affinity_transform: str = "aff1_potency",
    eps: float = 1e-9,
):
    # Summarize source observations by compound and protein.
    mol_df = canonicalize(mol_df, strict=False)

    # 1) CIDs dieta
    cid_cols = [c for c in diet_df.columns if c not in ["Codigo_dieta", "Diagnóstico"]]
    cid_ints = [int(c) for c in cid_cols]

    # 2) Labels
    mapped = diet_df["Diagnóstico"].map(LABEL_MAP)
    if mapped.isna().any():
        missing = diet_df.loc[mapped.isna(), "Diagnóstico"].unique()
        raise ValueError(f"Diagnóstico contiene valores no mapeados: {missing}")
    y = mapped.to_numpy()

    # 3) G y pesos
    grams = diet_df[cid_cols].to_numpy(dtype=np.float64)
    G = grams.sum(axis=1, keepdims=True)

    if weighting == "norm":
        W = grams / np.maximum(G, eps)
    elif weighting == "lognorm":
        grams_log = np.log1p(grams)
        denom = np.maximum(grams_log.sum(axis=1, keepdims=True), eps)
        W = grams_log / denom
    else:
        raise ValueError("weighting debe ser 'norm' o 'lognorm'")

    # 4) MolAgg
    desc_by_cid = (
        mol_df.groupby("CID")[DESC_COLS]
        .first()
        .reindex(cid_ints)
    )
    if desc_by_cid.isna().any().any():
        missing = desc_by_cid[desc_by_cid.isna().any(axis=1)].index.tolist()
        raise ValueError(f"Faltan descriptores SMILES para CIDs: {missing}")

    D = desc_by_cid.to_numpy(dtype=np.float64)
    MolAgg = W @ D

    # 5) ProtAgg
    # La tabla canonica ya posee una fila por CID-proteina.
    prot_table = (
        mol_df.groupby(["CID", "Proteina"])[PROT_COLS]
        .mean()
    )

    def matrix_for_protein(p: str) -> np.ndarray:
        mat = prot_table.xs(p, level="Proteina").reindex(cid_ints).fillna(0.0).copy()

        if affinity_transform == "aff1_potency":
            mat["Affinity1_rigid"] = np.maximum(0.0, -mat["Affinity1_rigid"])
        elif affinity_transform == "both_potency":
            mat["Affinity1_rigid"] = np.maximum(0.0, -mat["Affinity1_rigid"])
            mat["Affinity2_flex"] = np.maximum(0.0, -mat["Affinity2_flex"])
        elif affinity_transform == "none":
            pass
        else:
            raise ValueError(
                "affinity_transform debe ser 'none', 'aff1_potency' o 'both_potency'"
            )

        if prot_mode == "dock_only":
            return mat[["Affinity1_rigid"]].to_numpy(dtype=np.float64)

        return mat.to_numpy(dtype=np.float64)

    prot_blocks = []
    for p in PROTEINS:
        M = matrix_for_protein(p)
        prot_blocks.append(grams @ M)

    ProtAgg = np.concatenate(prot_blocks, axis=1)

    # 6) Combined
    X = np.concatenate([W, G, MolAgg, ProtAgg], axis=1).astype(np.float32)

    feat = []
    feat += [f"DietW__{c}" for c in cid_cols]
    feat += ["Diet__total_grams"]
    feat += [f"MolAgg__{c}" for c in DESC_COLS]

    if prot_mode == "dock_only":
        for p in PROTEINS:
            feat += [f"ProtAggDock__{p}__Affinity1"]
    else:
        for p in PROTEINS:
            feat += [f"ProtAgg__{p}__{c}" for c in PROT_COLS]

    return X, y.astype(int), feat


def main():
    if not MOL_PATH.is_file():
        raise FileNotFoundError(f"No existe: {MOL_PATH}")

    diet_path = next((p for p in DIET_CANDIDATES if p.is_file()), None)
    if diet_path is None:
        raise FileNotFoundError(
            "No encuentro base_dieta.csv. Puedes copiarlo junto a este script."
        )

    print(f"[OK] Dieta: {diet_path}")
    print(f"[OK] Molecular features: {MOL_PATH}")

    diet = pd.read_csv(diet_path)
    # Cast compound identifiers to integer before joining with dietary CIDs.
    mol = pd.read_csv(MOL_PATH)

    # Importante para que reindex(cid_ints) funcione como en el original.
    mol["CID"] = pd.to_numeric(mol["CID"], errors="raise").astype(int)

    X, y, feat = build_combined_features(
        diet,
        mol,
        weighting="norm",
        prot_mode="full",
        affinity_transform="aff1_potency",
    )

    if X.shape[1] != 106:
        raise RuntimeError(f"Esperaba 106 features y obtuve {X.shape[1]}.")

    if not np.isfinite(X).all():
        raise RuntimeError("X contiene NaN o infinitos.")

    out_dir = ROOT / "08_dataset_modelado_corregido"
    out_dir.mkdir(parents=True, exist_ok=True)

    out = pd.DataFrame(X, columns=feat)
    out.insert(0, "y", y)
    out.insert(0, "Diagnostico", diet["Diagnóstico"])
    out.insert(0, "Codigo_dieta", diet["Codigo_dieta"])

    out_path = out_dir / "combined_features_CORREGIDO.csv"
    out.to_csv(out_path, index=False)

    print("\n" + "=" * 78)
    print("RESULTADO")
    print("=" * 78)
    print(f"X: {X.shape}")
    print(f"y: {y.shape}")
    print(f"Class counts: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"NaN/inf: {not np.isfinite(X).all()}")
    print(f"Guardado: {out_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
