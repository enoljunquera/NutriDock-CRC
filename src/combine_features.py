import numpy as np
import pandas as pd

DIET_PATH = "base_dieta.csv"
MOL_PATH  = "molecular_features.csv"

# --- Config (como en tu paper) ---
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

PROTEINS = ["CYP1A1_4I8V","CYP1A2_2HI4","CYP2A4_1TQN","CYP2E1_3KOH"]  # orden fijo

LABEL_MAP = {"Normal": 0, "Pólipo": 1, "Neoplasia": 2}


def build_combined_features(
    diet_df: pd.DataFrame,
    mol_df: pd.DataFrame,
    weighting: str = "norm",          # "norm" o "lognorm"
    prot_mode: str = "full",          # "full" o "dock_only"
    affinity_transform: str = "aff1_potency",  # "none", "aff1_potency", "both_potency"
    eps: float = 1e-9,
):
    """
    Devuelve:
      X (n, 106) para prot_mode="full"
      X (n, 46)  para prot_mode="dock_only"
      y (n,)
      feature_names (list)
    """

    # --- 1) CIDs de dieta ---
    cid_cols = [c for c in diet_df.columns if c not in ["Codigo_dieta", "Diagnóstico"]]
    cid_ints = [int(c) for c in cid_cols]

    # --- 2) Labels ---
    y = diet_df["Diagnóstico"].map(LABEL_MAP).to_numpy()
    if np.any(pd.isna(diet_df["Diagnóstico"].map(LABEL_MAP))):
        missing = diet_df.loc[pd.isna(diet_df["Diagnóstico"].map(LABEL_MAP)), "Diagnóstico"].unique()
        raise ValueError(f"Diagnóstico contiene valores no mapeados: {missing}")

    # --- 3) G y pesos w ---
    grams = diet_df[cid_cols].to_numpy(dtype=np.float64)  # (n, 37)
    G = grams.sum(axis=1, keepdims=True)                  # (n, 1)

    if weighting == "norm":
        W = grams / np.maximum(G, eps)
    elif weighting == "lognorm":
        grams_log = np.log1p(grams)
        denom = np.maximum(grams_log.sum(axis=1, keepdims=True), eps)
        W = grams_log / denom
    else:
        raise ValueError("weighting debe ser 'norm' o 'lognorm'")

    # --- 4) Descriptores por CID (MolAgg) ---
    # OJO: mol_df trae una fila por proteína; aquí colapsamos a una por CID (primera)
    desc_by_cid = (
        mol_df.groupby("CID")[DESC_COLS]
        .first()
        .reindex(cid_ints)
    )
    if desc_by_cid.isna().any().any():
        missing = desc_by_cid[desc_by_cid.isna().any(axis=1)].index.tolist()
        raise ValueError(f"Faltan descriptores SMILES para CIDs: {missing}")

    D = desc_by_cid.to_numpy(dtype=np.float64)  # (37, 4)
    MolAgg = W @ D                               # (n, 4)

    # --- 5) ProtAgg: tabla CID–Proteina ---
    # En tu CSV hay duplicados CID–Proteina -> los colapsamos por media en columnas numéricas
    prot_table = (
        mol_df.groupby(["CID", "Proteina"])[PROT_COLS]
        .mean()
    )

    # Matrices por proteína, reindexadas a tus 37 CIDs (y rellenando faltantes con 0)
    def matrix_for_protein(p: str) -> np.ndarray:
        mat = prot_table.xs(p, level="Proteina").reindex(cid_ints).fillna(0.0).copy()

        # Transformaciones de afinidad
        if affinity_transform == "aff1_potency":
            mat["Affinity1_rigid"] = np.maximum(0.0, -mat["Affinity1_rigid"])
        elif affinity_transform == "both_potency":
            mat["Affinity1_rigid"] = np.maximum(0.0, -mat["Affinity1_rigid"])
            mat["Affinity2_flex"]  = np.maximum(0.0, -mat["Affinity2_flex"])
        elif affinity_transform == "none":
            pass
        else:
            raise ValueError("affinity_transform debe ser 'none', 'aff1_potency' o 'both_potency'")

        if prot_mode == "dock_only":
            # Solo una columna por proteína: docking burden (Affinity1 ya transformada si procede)
            return mat[["Affinity1_rigid"]].to_numpy(dtype=np.float64)  # (37, 1)

        return mat.to_numpy(dtype=np.float64)  # (37, 16)

    # Construimos ProtAgg por proteína: sum_c grams[c] * M[c, :]
    prot_blocks = []
    for p in PROTEINS:
        M = matrix_for_protein(p)         # (37, K)
        prot_blocks.append(grams @ M)     # (n, K)

    ProtAgg = np.concatenate(prot_blocks, axis=1)  # (n, 64) o (n, 4)

    # --- 6) Diet block + total grams ---
    # Diet: W (37) + G (1)
    # Combined: Diet(38) + MolAgg(4) + ProtAgg(64 o 4)
    X = np.concatenate([W, G, MolAgg, ProtAgg], axis=1).astype(np.float32)

    # Nombres de features
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


if __name__ == "__main__":
    diet = pd.read_csv(DIET_PATH)
    mol  = pd.read_csv(MOL_PATH)

    X, y, feat = build_combined_features(
        diet, mol,
        weighting="norm",
        prot_mode="full",
        affinity_transform="aff1_potency",
    )

    print("X:", X.shape, "y:", y.shape)
    print("Class counts:", dict(zip(*np.unique(y, return_counts=True))))

    # Guardar dataset final listo para modelar
    out = pd.DataFrame(X, columns=feat)
    out.insert(0, "y", y)
    out.insert(0, "Diagnostico", diet["Diagnóstico"])
    out.insert(0, "Codigo_dieta", diet["Codigo_dieta"])
    out.to_csv("combined_features.csv", index=False)
    print("Saved -> combined_features.csv")
