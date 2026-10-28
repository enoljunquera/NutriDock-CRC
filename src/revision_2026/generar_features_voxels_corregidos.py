#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extrae las features compactas de los voxels corregidos usando EXACTAMENTE
las funciones del conversor.py original.

Colocar este script y conversor.py en:
    06.FEEDBACK - Redocking\04_produccion_corregida

Entrada:
    05_voxels_corregidos\
        CYP1A1_4I8V\*.npy
        CYP1A2_2HI4\*.npy
        CYP2E1_3KOH\*.npy

Salida:
    06_features_voxels_corregidos\
        CYP1A1_4I8V\chem_voxel_features.csv/.parquet
        CYP1A2_2HI4\chem_voxel_features.csv/.parquet
        CYP2E1_3KOH\chem_voxel_features.csv/.parquet
"""

from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parent

try:
    import conversor
except ImportError:
    print("[ERROR] No encuentro conversor.py junto a este script.")
    print(f"        Colócalo en: {ROOT}")
    sys.exit(1)

PROTEINS = [
    "CYP1A1_4I8V",
    "CYP1A2_2HI4",
    "CYP2E1_3KOH",
]


def process_folder(label: str):
    in_dir = ROOT / "05_voxels_corregidos" / label
    out_dir = ROOT / "06_features_voxels_corregidos" / label
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 78)
    print(label)
    print("=" * 78)

    if not in_dir.is_dir():
        print(f"[ERROR] No existe: {in_dir}")
        return 0, 0

    npy_files = sorted(in_dir.glob("*.npy"))
    print(f"[INFO] NPY encontrados: {len(npy_files)}")

    if not npy_files:
        return 0, 0

    # EXACTAMENTE la misma definición que usaba conversor.py:
    masks = conversor.make_masks(size=25, core=9, mid=17)

    rows = []
    skipped = []

    for i, npy_path in enumerate(npy_files, start=1):
        try:
            row = conversor.build_features_from_npy(npy_path, masks)
            rows.append(row)
            print(f"[{i}/{len(npy_files)}] OK {npy_path.name}")
        except Exception as exc:
            skipped.append((npy_path.name, str(exc)))
            print(f"[{i}/{len(npy_files)}] SKIP {npy_path.name}: {exc}")

    if not rows:
        print("[ERROR] No se pudo procesar ningún NPY.")
        return 0, len(skipped)

    df = pd.DataFrame(rows)

    # Sanity checks del esquema original.
    expected_feature_cols = 48
    metadata_cols = {"chem_id", "target", "source_file"}
    feature_cols = [c for c in df.columns if c not in metadata_cols]

    if len(feature_cols) != expected_feature_cols:
        raise RuntimeError(
            f"Se esperaban {expected_feature_cols} features voxel crudas y "
            f"se obtuvieron {len(feature_cols)}."
        )

    csv_path = out_dir / "chem_voxel_features.csv"
    parquet_path = out_dir / "chem_voxel_features.parquet"

    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)

    print(f"\n[OK] Filas: {len(df)}")
    print(f"[OK] Columnas totales: {len(df.columns)} "
          f"(3 metadata + {len(feature_cols)} voxel)")
    print(f"[OK] CSV: {csv_path}")
    print(f"[OK] Parquet: {parquet_path}")

    if skipped:
        print(f"[AVISO] Saltados: {len(skipped)}")
        for name, err in skipped[:10]:
            print(f"        - {name}: {err}")

    return len(df), len(skipped)


def main():
    print("NutriDock-CRC | extracción de features voxel corregidas")
    print(f"Raíz: {ROOT}")
    print("Extractor: conversor.py ORIGINAL, sin modificar")

    total_ok = 0
    total_skip = 0

    for label in PROTEINS:
        ok, skip = process_folder(label)
        total_ok += ok
        total_skip += skip

    print("\n" + "=" * 78)
    print(f"TOTAL: filas generadas={total_ok} | saltadas={total_skip}")
    print(f"Salida: {ROOT / '06_features_voxels_corregidos'}")
    print("=" * 78)


if __name__ == "__main__":
    main()
