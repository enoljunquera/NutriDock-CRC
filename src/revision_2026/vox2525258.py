#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Genera un tensor 25x25x25x8 a partir de:
  - un receptor rígido (PDBQT)
  - un ligando en pose de docking flexible (PDBQT)

Salida: fichero .npy con shape (25, 25, 25, 8)

Canales por voxel:
  0: G1 (Gauss1)
  1: G2 (Gauss2)
  2: R  (Repulsión)
  3: H  (Hidrofobicidad)
  4: AC (H-bond Acceptor, booleano)
  5: DC (H-bond Donor, booleano)
  6: NH (Non-directional H-bond, booleano)
  7: M  (Máscara de interacción, booleano)

NOTA:
- Las fórmulas de energía están basadas en el esquema de AutoDock Vina.
- Los parámetros (w, alpha, mu, etc.) están como placeholders típicos.
  Si quieres reproducir EXACTO tu artículo, cámbialos por tus valores.
"""

import os
import math
import argparse
from typing import List, Dict, Tuple

import numpy as np

# ==========================
# 1. PARÁMETROS GLOBALES
# ==========================

GRID_SIZE = 25           # 25x25x25
BOX_SIZE  = 25.0         # lado del cubo en Å (AJUSTA A LO QUE USASTE)
VOXEL_SIZE = BOX_SIZE / GRID_SIZE

# Parámetros aproximados estilo Vina (cámbialos si quieres precisión exacta)
VINA_PARAMS = {
    # G(r) = w * exp(-alpha (r - mu)^2)
    "G1": {"w": -0.035579, "alpha": 0.5, "mu": 0.0},
    "G2": {"w": -0.005156, "alpha": 0.5, "mu": 3.0},
    # Repulsión ~ w * r^-12
    "R": {"w": 0.840245},
    # Hidrofobicidad: placeholder sencillo
    "H": {"w": -0.035069, "cutoff": 8.0}
}

# Radio máximo para considerar interacciones y activar la máscara M (Å)
MASK_RADIUS = 12.0


# ==========================
# 2. PARSEO DE PDBQT
# ==========================

def parse_pdbqt_atoms(path: str,
                      keep_hydrogens: bool = False) -> List[Dict]:
    """
    Parsea ATOM/HETATM de un PDBQT de Vina.
    Devuelve lista de dicts con:
      - x, y, z (coordenadas en Å)
      - element (C, N, O...)
      - vina_type (tipo de Vina, último campo de la línea)
    """
    atoms = []
    with open(path, "r") as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue

            name = line[12:16].strip()
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])

            element = line[76:78].strip()
            parts = line.split()
            vina_type = parts[-1]  # último token suele ser el tipo PDBQT

            if not element:
                element = name[0]  # fallback

            if not keep_hydrogens and element.upper() == "H":
                continue

            atoms.append({
                "name": name,
                "x": x,
                "y": y,
                "z": z,
                "element": element,
                "vina_type": vina_type
            })
    return atoms


def ligand_center(ligand_atoms: List[Dict]) -> np.ndarray:
    xs = [a["x"] for a in ligand_atoms]
    ys = [a["y"] for a in ligand_atoms]
    zs = [a["z"] for a in ligand_atoms]
    return np.array([np.mean(xs), np.mean(ys), np.mean(zs)], dtype=np.float32)


def world_to_grid(coord: np.ndarray,
                  center: np.ndarray) -> Tuple[int, int, int]:
    """
    Convierte coordenadas (x,y,z) en Å → índices (i,j,k) del voxel 0..24.

    El cubo se centra en 'center' (centro del ligando) y tiene lado BOX_SIZE.
    """
    min_corner = center - BOX_SIZE / 2.0
    rel = coord - min_corner
    idx = rel / VOXEL_SIZE

    i = int(math.floor(idx[0]))
    j = int(math.floor(idx[1]))
    k = int(math.floor(idx[2]))

    if 0 <= i < GRID_SIZE and 0 <= j < GRID_SIZE and 0 <= k < GRID_SIZE:
        return i, j, k
    else:
        return -1, -1, -1  # fuera del cubo


# ==========================
# 3. TÉRMINOS ENERGÉTICOS
# ==========================

def gauss_term(r: float, w: float, alpha: float, mu: float) -> float:
    return w * math.exp(-alpha * (r - mu) ** 2)


def repulsion_term(r: float, w: float) -> float:
    if r < 1e-6:
        r = 1e-6
    return w * (r ** -12)


def hydrophobic_term(r: float, w: float, cutoff: float) -> float:
    if r > cutoff:
        return 0.0
    # decaimiento lineal muy simple
    return w * (1.0 - r / cutoff)


# ==========================
# 4. CANALES BOOLEANOS
# ==========================

def classify_booleans_for_atom(atom: Dict) -> Tuple[int, int, int, bool]:
    """
    Clasificación heurística basada en tipo de Vina + elemento.
    Devuelve:
      (is_acceptor, is_donor, is_nh, is_hydrophobic_candidate)

    Para reproducir EXACTO tu paper:
      - sustituir por lógica basada en RDKit + SMARTS.
    """
    elem = atom["element"].upper()
    atype = atom["vina_type"].upper()

    is_acceptor = 0
    is_donor = 0
    is_nh = 0
    is_hydrophobic = False

    # muy simplificado:
    # aceptores típicos: OA, NA
    if "OA" in atype or "NA" in atype:
        is_acceptor = 1
        is_nh = 1

    # donores típicos: HD
    if "HD" in atype:
        is_donor = 1
        is_nh = 1

    # hidrofóbicos: carbonos apolares (heurístico)
    if elem == "C" and ("A" not in atype):
        is_hydrophobic = True

    return is_acceptor, is_donor, is_nh, is_hydrophobic


# ==========================
# 5. VOXELIZACIÓN
# ==========================

def voxelize_complex(receptor_atoms: List[Dict],
                     ligand_atoms: List[Dict]) -> np.ndarray:
    """
    Devuelve un tensor (25,25,25,8) con canales:
      [G1, G2, R, H, AC, DC, NH, M]
    """
    grid = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE, 8), dtype=np.float32)

    center = ligand_center(ligand_atoms)

    # Booleanos para cada átomo del ligando
    ligand_bool = []
    for a in ligand_atoms:
        ac, dc, nh, is_hydrophobic = classify_booleans_for_atom(a)
        ligand_bool.append((ac, dc, nh, is_hydrophobic))

    # 5.1 Interacciones ligando–receptor → energías
    for li, latom in enumerate(ligand_atoms):
        la = np.array([latom["x"], latom["y"], latom["z"]], dtype=np.float32)

        for ratom in receptor_atoms:
            ra = np.array([ratom["x"], ratom["y"], ratom["z"]], dtype=np.float32)

            diff = la - ra
            r = float(np.linalg.norm(diff))
            if r > MASK_RADIUS:
                continue

            midpoint = (la + ra) / 2.0
            i, j, k = world_to_grid(midpoint, center)
            if i < 0:
                continue

            G1 = gauss_term(r, **VINA_PARAMS["G1"])
            G2 = gauss_term(r, **VINA_PARAMS["G2"])
            R  = repulsion_term(r, VINA_PARAMS["R"]["w"])

            _, _, _, lig_is_hydrophobic = ligand_bool[li]
            if lig_is_hydrophobic:
                H = hydrophobic_term(r,
                                     VINA_PARAMS["H"]["w"],
                                     VINA_PARAMS["H"]["cutoff"])
            else:
                H = 0.0

            grid[i, j, k, 0] += G1
            grid[i, j, k, 1] += G2
            grid[i, j, k, 2] += R
            grid[i, j, k, 3] += H

            # máscara de interacción
            grid[i, j, k, 7] = 1.0

    # 5.2 Canales booleanos por átomo de ligando
    for li, latom in enumerate(ligand_atoms):
        la = np.array([latom["x"], latom["y"], latom["z"]], dtype=np.float32)
        i, j, k = world_to_grid(la, center)
        if i < 0:
            continue

        ac, dc, nh, _ = ligand_bool[li]

        if ac:
            grid[i, j, k, 4] = 1.0
        if dc:
            grid[i, j, k, 5] = 1.0
        if nh:
            grid[i, j, k, 6] = 1.0

        # también activamos máscara donde hay átomo de ligando
        grid[i, j, k, 7] = 1.0

    return grid


# ==========================
# 6. MAIN
# ==========================

def main():
    parser = argparse.ArgumentParser(
        description="Voxelización 25x25x25x8 de un complejo receptor–ligando (PDBQT)"
    )
    parser.add_argument("--receptor", required=True,
                        help="PDBQT del receptor rígido")
    parser.add_argument("--ligand", required=True,
                        help="PDBQT del ligando en la pose de docking flexible")
    parser.add_argument("--out", required=True,
                        help="Fichero de salida .npy (tensor 25x25x25x8)")

    args = parser.parse_args()

    if not os.path.isfile(args.receptor):
        raise FileNotFoundError(f"No se encuentra el receptor: {args.receptor}")
    if not os.path.isfile(args.ligand):
        raise FileNotFoundError(f"No se encuentra el ligando: {args.ligand}")

    print(f"[INFO] Leyendo receptor: {args.receptor}")
    receptor_atoms = parse_pdbqt_atoms(args.receptor, keep_hydrogens=False)
    print(f"[INFO] Átomos receptor: {len(receptor_atoms)}")

    print(f"[INFO] Leyendo ligando: {args.ligand}")
    ligand_atoms = parse_pdbqt_atoms(args.ligand, keep_hydrogens=False)
    print(f"[INFO] Átomos ligando: {len(ligand_atoms)}")

    if not receptor_atoms or not ligand_atoms:
        raise RuntimeError("Receptor o ligando sin átomos válidos (tras eliminar H).")

    print("[INFO] Voxelizando complejo...")
    vox = voxelize_complex(receptor_atoms, ligand_atoms)
    print(f"[INFO] Tensor generado con shape {vox.shape}")

    np.save(args.out, vox)
    print(f"[OK] Guardado: {args.out}")


if __name__ == "__main__":
    main()
