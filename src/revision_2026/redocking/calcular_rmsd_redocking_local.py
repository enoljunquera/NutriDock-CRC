from pathlib import Path
import math
import re
import csv

ROOT = Path(__file__).resolve().parent

SYSTEMS = {
    "CYP1A1_4I8V_BHF": {
        "dock": "redock_local_BHF_4I8V.pdbqt",
        "refs": "crystal_BHF_*.pdb",
    },
    "CYP1A2_2HI4_BHF": {
        "dock": "redock_local_BHF_2HI4.pdbqt",
        "refs": "crystal_BHF_*.pdb",
    },
    "CYP2E1_3KOH_OIO": {
        "dock": "redock_local_OIO_3KOH.pdbqt",
        "refs": "crystal_OIO_*.pdb",
    },
    "CYP3A4_3NXU_RIT": {
        "dock": "redock_local_RIT_3NXU.pdbqt",
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

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue

            element = pdb_element(line)

            if element == "H":
                continue

            name = line[12:16].strip()

            if name in atoms:
                raise RuntimeError(
                    f"{path.name}: átomo duplicado {name}"
                )

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

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
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

                if name in current["atoms"]:
                    raise RuntimeError(
                        f"{path.name}: átomo duplicado "
                        f"{name} en modo {current['mode']}"
                    )

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
            "\nCorrespondencia de átomos incompatible.\n"
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
summary = []

print("=" * 88)
print("NutriDock-CRC | RMSD de redocking LOCAL")
print("=" * 88)

for system, cfg in SYSTEMS.items():

    folder = ROOT / system
    dock_path = folder / cfg["dock"]
    refs = sorted(folder.glob(cfg["refs"]))

    if not dock_path.is_file():
        raise FileNotFoundError(dock_path)

    if not refs:
        raise FileNotFoundError(
            f"{system}: no encuentro referencias {cfg['refs']}"
        )

    models = read_vina_models(dock_path)

    print("\n" + "=" * 88)
    print(system)
    print("=" * 88)
    print(f"Archivo docking              : {dock_path.name}")
    print(f"Poses Vina encontradas       : {len(models)}")
    print(f"Referencias cristalográficas : {len(refs)}")
    print()

    best_system = None
    mode1_result = None

    for model in models:

        best_for_mode = None

        for ref in refs:
            crystal = read_crystal(ref)

            value = rmsd(
                model["atoms"],
                crystal,
            )

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

        if model["mode"] == 1:
            mode1_result = (
                value,
                model["affinity"],
                ref_name,
            )

        if best_system is None or value < best_system[0]:
            best_system = (
                value,
                model["mode"],
                model["affinity"],
                ref_name,
            )

    if mode1_result is None:
        raise RuntimeError(
            f"{system}: no encuentro modo 1."
        )

    top_rmsd, top_affinity, top_ref = mode1_result
    best_rmsd, best_mode, best_affinity, best_ref = best_system

    print("\nRESULTADO DEL SISTEMA")
    print(f"Top-ranked (modo 1) RMSD : {top_rmsd:.3f} Å")
    print(f"Top-ranked referencia    : {top_ref}")
    print(f"Mejor RMSD entre modos   : {best_rmsd:.3f} Å")
    print(f"Mejor modo por RMSD      : {best_mode}")
    print(f"Mejor referencia         : {best_ref}")

    top_pass = top_rmsd <= 2.0
    sampling_pass = best_rmsd <= 2.0

    print(
        "Pose top-ranked <= 2 Å     : "
        + ("SÍ" if top_pass else "NO")
    )
    print(
        "Alguna pose <= 2 Å         : "
        + ("SÍ" if sampling_pass else "NO")
    )

    summary.append({
        "system": system,
        "top_rmsd_A": top_rmsd,
        "top_affinity_kcal_mol": top_affinity,
        "top_reference": top_ref,
        "best_rmsd_A": best_rmsd,
        "best_mode": best_mode,
        "best_affinity_kcal_mol": best_affinity,
        "best_reference": best_ref,
        "top_pose_le_2A": top_pass,
        "any_pose_le_2A": sampling_pass,
    })


out_all = ROOT / "rmsd_redocking_local_todos_modos.csv"

with open(out_all, "w", newline="", encoding="utf-8") as f:
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


out_summary = ROOT / "rmsd_redocking_local_resumen.csv"

with open(out_summary, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "system",
            "top_rmsd_A",
            "top_affinity_kcal_mol",
            "top_reference",
            "best_rmsd_A",
            "best_mode",
            "best_affinity_kcal_mol",
            "best_reference",
            "top_pose_le_2A",
            "any_pose_le_2A",
        ],
    )
    writer.writeheader()
    writer.writerows(summary)


print("\n" + "=" * 88)
print("RESUMEN GLOBAL")
print("=" * 88)

for row in summary:
    print(
        f"{row['system']:<24} "
        f"top={row['top_rmsd_A']:>7.3f} Å | "
        f"best={row['best_rmsd_A']:>7.3f} Å "
        f"(modo {row['best_mode']}) | "
        f"top<=2: {'SÍ' if row['top_pose_le_2A'] else 'NO'} | "
        f"any<=2: {'SÍ' if row['any_pose_le_2A'] else 'NO'}"
    )

print()
print(f"CSV todos los modos : {out_all.name}")
print(f"CSV resumen         : {out_summary.name}")
print("=" * 88)