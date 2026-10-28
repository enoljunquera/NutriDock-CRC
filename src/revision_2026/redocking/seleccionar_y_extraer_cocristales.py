from pathlib import Path
from collections import defaultdict
import math
import re

ROOT = Path(__file__).resolve().parent

SYSTEMS = {
    "CYP1A1_4I8V_BHF": {
        "ligand": "BHF",
        "center": (-31.343, 78.495, -19.031),
    },
    "CYP1A2_2HI4_BHF": {
        "ligand": "BHF",
        "center": (7.288, 21.552, 25.238),
    },
    "CYP2E1_3KOH_OIO": {
        "ligand": "OIO",
        "center": (15.883, 17.808, -4.758),
    },
}


def find_original_pdb(folder):
    pdbs = [
        p for p in folder.glob("*.pdb")
        if "crystal_" not in p.name.lower()
        and "clean" not in p.name.lower()
    ]

    if len(pdbs) != 1:
        raise RuntimeError(
            f"{folder.name}: esperaba 1 PDB original y encontré {len(pdbs)}"
        )

    return pdbs[0]


def atom_xyz(line):
    return (
        float(line[30:38]),
        float(line[38:46]),
        float(line[46:54]),
    )


def centroid(lines):
    xyz = [atom_xyz(line) for line in lines]

    return (
        sum(p[0] for p in xyz) / len(xyz),
        sum(p[1] for p in xyz) / len(xyz),
        sum(p[2] for p in xyz) / len(xyz),
    )


def distance(a, b):
    return math.sqrt(
        (a[0] - b[0]) ** 2
        + (a[1] - b[1]) ** 2
        + (a[2] - b[2]) ** 2
    )


def atom_serial(line):
    try:
        return int(line[6:11])
    except ValueError:
        return None


for folder_name, cfg in SYSTEMS.items():

    folder = ROOT / folder_name
    ligand = cfg["ligand"]
    dock_center = cfg["center"]

    print("\n" + "=" * 72)
    print(folder_name)
    print("=" * 72)

    pdb_file = find_original_pdb(folder)

    with open(pdb_file, "r", encoding="utf-8", errors="ignore") as f:
        all_lines = f.readlines()

    copies = defaultdict(list)

    for line in all_lines:
        if not line.startswith("HETATM"):
            continue

        resname = line[17:20].strip()
        if resname != ligand:
            continue

        chain = line[21].strip() or "-"
        resseq = line[22:26].strip()
        icode = line[26].strip() or "-"

        copies[(chain, resseq, icode)].append(line)

    if not copies:
        raise RuntimeError(
            f"{folder_name}: no se encontró {ligand}"
        )

    ranked = []

    print(f"Centro docking: {dock_center}")
    print()

    for key, lines in copies.items():
        c = centroid(lines)
        d = distance(c, dock_center)

        ranked.append((d, key, c, lines))

        chain, resseq, icode = key

        print(
            f"Cadena={chain:>2} | Residuo={resseq:>4} | "
            f"centro=({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}) | "
            f"distancia al centro docking={d:.3f} Å"
        )

    ranked.sort(key=lambda x: x[0])

    best_distance, best_key, best_centroid, best_lines = ranked[0]

    chain, resseq, icode = best_key

    output = folder / f"crystal_{ligand}_{chain}_{resseq}.pdb"

    selected_serials = {
        atom_serial(line)
        for line in best_lines
        if atom_serial(line) is not None
    }

    with open(output, "w", encoding="utf-8") as out:

        out.write(
            f"REMARK Selected co-crystallized ligand {ligand}\n"
        )
        out.write(
            f"REMARK Chain {chain} residue {resseq}\n"
        )
        out.write(
            f"REMARK Distance from ligand centroid to production docking "
            f"center: {best_distance:.3f} Angstrom\n"
        )

        for line in best_lines:
            out.write(line)

        # Conservar CONECT únicamente cuando involucra átomos
        # pertenecientes al ligando seleccionado.
        for line in all_lines:
            if not line.startswith("CONECT"):
                continue

            serials = [
                int(x)
                for x in re.findall(r"\d+", line[6:])
            ]

            if serials and serials[0] in selected_serials:
                out.write(line)

        out.write("END\n")

    print()
    print(
        f"SELECCIONADO: {ligand} cadena {chain}, "
        f"residuo {resseq}"
    )
    print(
        f"Distancia: {best_distance:.3f} Å"
    )
    print(
        f"Guardado: {output.name}"
    )

print("\n" + "=" * 72)
print("Extracción completada para los tres sistemas.")
print("=" * 72)