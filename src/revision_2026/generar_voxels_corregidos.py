#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Genera los voxels corregidos para CYP1A1/4I8V, CYP1A2/2HI4 y CYP2E1/3KOH
a partir de los dockings flexibles corregidos.

IMPORTANTE:
- Usa SIN MODIFICAR las funciones de vox2525258.py.
- Usa el receptor rígido específico de cada pareja generado por MGLTools.
- Usa la pose final del docking flexible.
- Omite gen3d.
- Omite salidas Vina vacías o sin REMARK VINA RESULT.
- No toca los voxels antiguos.
- CYP3A4/1TQN no se recalcula porque no estaba afectado por la contaminación
  del receptor cristalográfico.

Colocar este script y vox2525258.py en:
    06.FEEDBACK - Redocking\04_produccion_corregida

Ejecutar:
    python generar_voxels_corregidos.py
"""

from pathlib import Path
import re
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent

try:
    import vox2525258 as vox
except ImportError:
    print("[ERROR] No encuentro vox2525258.py junto a este script.")
    print(f"        Colócalo en: {ROOT}")
    sys.exit(1)


PROTEINS = [
    {
        "folder": "CYP1A1_4I8V",
        "code": "4I8V",
        "label": "CYP1A1_4I8V",
        "receptor_base": "CYP1A1_4I8V_clean",
    },
    {
        "folder": "CYP1A2_2HI4",
        "code": "2HI4",
        "label": "CYP1A2_2HI4",
        "receptor_base": "CYP1A2_2HI4_clean",
    },
    {
        "folder": "CYP2E1_3KOH",
        "code": "3KOH",
        "label": "CYP2E1_3KOH",
        "receptor_base": "CYP2E1_3KOH_clean",
    },
]


def has_vina_result(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            return any("REMARK VINA RESULT:" in line for line in f)
    except OSError:
        return False


def canonical_ligand_id(ligand_tag: str) -> str:
    """
    ligand_CaffeicacydCID689043 -> CaffeicacydCID689043
    ligand_PallidolCID484757CID484757 -> PallidolCID484757

    Se conserva el nombre actual hasta el PRIMER CID, porque el emparejamiento
    posterior puede hacerse de forma inequívoca por CID.
    """
    s = ligand_tag
    if s.lower().startswith("ligand_"):
        s = s[len("ligand_"):]

    m = re.match(r"(.+?CID\d+)", s, flags=re.IGNORECASE)
    return m.group(1) if m else s


def process_protein(cfg):
    folder = cfg["folder"]
    code = cfg["code"]
    label = cfg["label"]
    receptor_base = cfg["receptor_base"]

    flex_dir = ROOT / folder / f"flex_{code}"
    pose_dir = flex_dir / "results" / "poses"

    # Salida separada de la producción antigua.
    out_dir = ROOT / "05_voxels_corregidos" / label
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 78)
    print(label)
    print("=" * 78)

    if not pose_dir.is_dir():
        print(f"[ERROR] No existe: {pose_dir}")
        return 0, 0, 0

    poses = sorted(pose_dir.glob(f"dock_{code}_*_flex.pdbqt"))
    print(f"[INFO] Poses flexibles encontradas: {len(poses)}")

    ok = 0
    skipped_invalid = 0
    skipped_missing_receptor = 0

    for i, pose in enumerate(poses, start=1):
        stem = pose.stem
        prefix = f"dock_{code}_"
        suffix = "_flex"

        if not stem.startswith(prefix) or not stem.endswith(suffix):
            continue

        ligand_tag = stem[len(prefix):-len(suffix)]

        if "gen3d" in ligand_tag.lower():
            print(f"[{i}/{len(poses)}] {ligand_tag}: SKIP gen3d")
            continue

        if not has_vina_result(pose):
            print(f"[{i}/{len(poses)}] {ligand_tag}: SKIP pose vacía/no válida")
            skipped_invalid += 1
            continue

        rigid = flex_dir / f"{receptor_base}_rigid_{ligand_tag}.pdbqt"
        if not rigid.is_file() or rigid.stat().st_size == 0:
            print(f"[{i}/{len(poses)}] {ligand_tag}: SKIP receptor rígido no encontrado")
            skipped_missing_receptor += 1
            continue

        ligand_id = canonical_ligand_id(ligand_tag)
        out_path = out_dir / f"{label}_{ligand_id}_25x25x25x8.npy"

        print(f"[{i}/{len(poses)}] {ligand_tag}")
        print(f"      receptor: {rigid.name}")
        print(f"      pose    : {pose.name}")

        receptor_atoms = vox.parse_pdbqt_atoms(str(rigid), keep_hydrogens=False)
        ligand_atoms = vox.parse_pdbqt_atoms(str(pose), keep_hydrogens=False)

        if not receptor_atoms or not ligand_atoms:
            print("      [SKIP] receptor o pose sin átomos válidos")
            skipped_invalid += 1
            continue

        tensor = vox.voxelize_complex(receptor_atoms, ligand_atoms)

        if tensor.shape != (25, 25, 25, 8):
            print(f"      [ERROR] shape inesperado: {tensor.shape}")
            skipped_invalid += 1
            continue

        np.save(out_path, tensor)
        print(f"      [OK] {out_path.name}")
        ok += 1

    print(
        f"\n{label}: generados={ok} | "
        f"pose_invalida={skipped_invalid} | "
        f"sin_receptor={skipped_missing_receptor}"
    )
    return ok, skipped_invalid, skipped_missing_receptor


def main():
    print("NutriDock-CRC | regeneración de voxels corregidos")
    print(f"Raíz: {ROOT}")
    print("Algoritmo de voxelización: vox2525258.py SIN MODIFICAR")

    total_ok = 0
    total_invalid = 0
    total_missing = 0

    for cfg in PROTEINS:
        ok, invalid, missing = process_protein(cfg)
        total_ok += ok
        total_invalid += invalid
        total_missing += missing

    print("\n" + "=" * 78)
    print(
        f"TOTAL: generados={total_ok} | "
        f"poses inválidas/ausentes={total_invalid} | "
        f"receptores rígidos ausentes={total_missing}"
    )
    print(f"Salida: {ROOT / '05_voxels_corregidos'}")
    print("=" * 78)


if __name__ == "__main__":
    main()
