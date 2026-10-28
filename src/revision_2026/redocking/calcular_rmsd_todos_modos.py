from pathlib import Path
import math
import re
import csv

ROOT = Path(__file__).resolve().parent

SYSTEMS = {
    "CYP1A1_4I8V_BHF": {
        "dock": "redock_rigid_BHF_4I8V.pdbqt",
        "refs": "crystal_BHF_*.pdb",
    },
    "CYP1A2_2HI4_BHF": {
        "dock": "redock_rigid_BHF_2HI4.pdbqt",
        "refs": "crystal_BHF_*.pdb",
    },
    "CYP2E1_3KOH_OIO": {
        "dock": "redock_rigid_OIO_3KOH.pdbqt",
        "refs": "crystal_OIO_*.pdb",
    },
    "CYP3A4_3NXU_RIT": {
        "dock": "redock_rigid_RIT_3NXU.pdbqt",
        "refs": "crystal_RIT_*.pdb",
    },
}


def pdb_element(line):
    if len(line) >= 78:
        element = line[76:78].strip()
        if element:
            return element.upper()

    name = line[12:16].strip()
    name = re.sub(r"^[0-9]+", "", name)

    if len(name) >= 2 and name[:2].upper() in {"CL", "BR"}:
        return name[:2].upper()

    return name[0].upper() if name else ""


def pdbqt_element(line):
    parts = line.split()
    ad_type = parts[-1].upper()

    mapping = {
        "A": "C",
        "C": "C",
        "N": "N",
        "NA": "N",
        "NS": "N",
        "OA": "O",
        "OS": "O",
        "O": "O",
        "S": "S",
        "SA": "S",
        "P": "P",
        "F": "F",
        "CL": "CL",
        "BR": "BR",
        "I": "I",
        "H": "H",
        "HD": "H",
        "HS": "H",
    }

    return mapping.get(ad_type, ad_type)


def read_crystal(path):
    atoms = {}

    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue

            element = pdb_element(line)
            if element == "H":
                continue

            name = line[12:16].strip()

            atoms[name] = {
                "element": element,
                "coord": (
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                ),
            }

    return atoms


def read_vina_models(path):
    models = []

    current = None

    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:

            if line.startswith("MODEL"):
                if current is not None:
                    models.append(current)

                try:
                    mode = int(line.split()[1])
                except Exception:
                    mode = len(models) + 1

                current = {
                    "mode": mode,
                    "affinity": None,
                    "atoms": {},
                }
                continue

            if current is None:
                continue

            if line.startswith("REMARK VINA RESULT:"):
                parts = line.split()
                current["affinity"] = float(parts[3])
                continue

            if line.startswith(("ATOM  ", "HETATM")):
                element = pdbqt_element(line)

                if element == "H":
                    continue

                name = line[12:16].strip()

                current["atoms"][name] = {
                    "element": element,
                    "coord": (
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    ),
                }

            if line.startswith("ENDMDL"):
                models.append(current)
                current = None

    if current is not None:
        models.append(current)

    return models


def rmsd(docked, crystal):

    if set(docked) != set(crystal):
        raise RuntimeError(
            f"Correspondencia de átomos incompatible.\n"
            f"Docking: {sorted(docked)}\n"
            f"Cristal: {sorted(crystal)}"
        )

    total = 0.0

    for name in docked:

        if docked[name]["element"] != crystal[name]["element"]:
            raise RuntimeError(
                f"Elemento distinto para {name}: "
                f"{docked[name]['element']} vs "
                f"{crystal[name]['element']}"
            )

        x1, y1, z1 = docked[name]["coord"]
        x2, y2, z2 = crystal[name]["coord"]

        total += (
            (x1 - x2) ** 2
            + (y1 - y2) ** 2
            + (z1 - z2) ** 2
        )

    return math.sqrt(total / len(docked))


rows = []

print("=" * 84)
print("NutriDock-CRC | RMSD de todos los modos de redocking")
print("=" * 84)

for system, cfg in SYSTEMS.items():

    folder = ROOT / system
    dock_path = folder / cfg["dock"]
    refs = sorted(folder.glob(cfg["refs"]))

    models = read_vina_models(dock_path)

    print("\n" + "=" * 84)
    print(system)
    print("=" * 84)
    print(f"Poses Vina encontradas: {len(models)}")
    print(f"Referencias cristalográficas: {len(refs)}\n")

    best_system = None

    for model in models:

        best_for_mode = None

        for ref in refs:
            crystal = read_crystal(ref)

            value = rmsd(model["atoms"], crystal)

            candidate = (value, ref.name)

            if best_for_mode is None or candidate[0] < best_for_mode[0]:
                best_for_mode = candidate

        value, ref_name = best_for_mode

        print(
            f"Modo {model['mode']:>2} | "
            f"afinidad {model['affinity']:>6.2f} kcal/mol | "
            f"RMSD mínimo {value:>8.3f} Å | "
            f"{ref_name}"
        )

        rows.append({
            "system": system,
            "mode": model["mode"],
            "affinity_kcal_mol": model["affinity"],
            "best_reference": ref_name,
            "rmsd_A": value,
        })

        if best_system is None or value < best_system[0]:
            best_system = (
                value,
                model["mode"],
                model["affinity"],
                ref_name,
            )

    value, mode, affinity, ref_name = best_system

    print("\nMEJOR POSE POR RMSD")
    print(f"Modo       : {mode}")
    print(f"Afinidad   : {affinity:.2f} kcal/mol")
    print(f"Referencia : {ref_name}")
    print(f"RMSD       : {value:.3f} Å")


out = ROOT / "rmsd_redocking_todos_modos.csv"

with open(out, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "system",
            "mode",
            "affinity_kcal_mol",
            "best_reference",
            "rmsd_A",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)


print("\n" + "=" * 84)
print(f"CSV guardado: {out.name}")
print("=" * 84)