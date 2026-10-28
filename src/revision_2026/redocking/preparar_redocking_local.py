from pathlib import Path

ROOT = Path(__file__).resolve().parent

SYSTEMS = {
    "CYP1A1_4I8V_BHF": {
        "receptor": "CYP1A1_4I8V_clean.pdbqt",
        "ligand": "redock_BHF_4I8V.pdbqt",
        "crystal": "crystal_BHF_B_602.pdb",
        "out": "redock_local_BHF_4I8V.pdbqt",
    },
    "CYP1A2_2HI4_BHF": {
        "receptor": "CYP1A2_2HI4_clean.pdbqt",
        "ligand": "redock_BHF_2HI4.pdbqt",
        "crystal": "crystal_BHF_A_800.pdb",
        "out": "redock_local_BHF_2HI4.pdbqt",
    },
    "CYP2E1_3KOH_OIO": {
        "receptor": "CYP2E1_3KOH_clean.pdbqt",
        "ligand": "redock_OIO_3KOH.pdbqt",
        "crystal": "crystal_OIO_B_1.pdb",
        "out": "redock_local_OIO_3KOH.pdbqt",
    },
    "CYP3A4_3NXU_RIT": {
        "receptor": "CYP3A4_3NXU_chainA_clean_HEM.pdbqt",
        "ligand": "redock_RIT_3NXU.pdbqt",
        "crystal": "crystal_RIT_A_600.pdb",
        "out": "redock_local_RIT_3NXU.pdbqt",
    },
}

MARGIN = 10.0

EXHAUSTIVENESS = 24
NUM_MODES = 9
ENERGY_RANGE = 3


def read_pdb_coords(path):
    atoms = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue

            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue

            atoms.append((x, y, z))

    return atoms


def read_pdbqt_coords(path):
    atoms = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue

            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue

            atoms.append((x, y, z))

    return atoms


def make_box(coords, margin):
    xs = [a[0] for a in coords]
    ys = [a[1] for a in coords]
    zs = [a[2] for a in coords]

    min_xyz = (min(xs), min(ys), min(zs))
    max_xyz = (max(xs), max(ys), max(zs))

    center = tuple(
        (lo + hi) / 2.0
        for lo, hi in zip(min_xyz, max_xyz)
    )

    size = tuple(
        (hi - lo) + 2.0 * margin
        for lo, hi in zip(min_xyz, max_xyz)
    )

    return center, size


def inside_box(atom, center, size):
    return all(
        abs(a - c) <= s / 2.0 + 1e-6
        for a, c, s in zip(atom, center, size)
    )


print("=" * 78)
print("NutriDock-CRC | preparación de redocking LOCAL")
print("=" * 78)

for system, cfg in SYSTEMS.items():

    folder = ROOT / system

    receptor = folder / cfg["receptor"]
    ligand = folder / cfg["ligand"]
    crystal = folder / cfg["crystal"]
    conf = folder / "conf_redock_local.txt"

    for path in (receptor, ligand, crystal):
        if not path.is_file():
            raise FileNotFoundError(path)

    crystal_atoms = read_pdb_coords(crystal)
    ligand_atoms = read_pdbqt_coords(ligand)

    if not crystal_atoms:
        raise RuntimeError(
            f"{system}: no encuentro átomos en {crystal.name}"
        )

    center, size = make_box(crystal_atoms, MARGIN)

    outside = [
        atom for atom in ligand_atoms
        if not inside_box(atom, center, size)
    ]

    print("\n" + "=" * 78)
    print(system)
    print("=" * 78)

    print(f"Referencia cristalográfica : {crystal.name}")
    print(f"Átomos cristal             : {len(crystal_atoms)}")
    print(f"Átomos ligando PDBQT       : {len(ligand_atoms)}")

    print(
        "\nCentro:\n"
        f"  X = {center[0]:.3f}\n"
        f"  Y = {center[1]:.3f}\n"
        f"  Z = {center[2]:.3f}"
    )

    print(
        "\nTamaño:\n"
        f"  X = {size[0]:.3f}\n"
        f"  Y = {size[1]:.3f}\n"
        f"  Z = {size[2]:.3f}"
    )

    if outside:
        print(
            f"\n[ERROR] {len(outside)} átomos del ligando "
            "PDBQT quedan fuera de la caja."
        )
        raise SystemExit(1)

    print("\nAUDITORÍA: todo el ligando queda dentro de la caja.")

    with open(conf, "w", encoding="utf-8") as f:
        f.write(f"receptor = {receptor.name}\n")
        f.write(f"ligand = {ligand.name}\n\n")

        f.write(f"center_x = {center[0]:.3f}\n")
        f.write(f"center_y = {center[1]:.3f}\n")
        f.write(f"center_z = {center[2]:.3f}\n\n")

        f.write(f"size_x = {size[0]:.3f}\n")
        f.write(f"size_y = {size[1]:.3f}\n")
        f.write(f"size_z = {size[2]:.3f}\n\n")

        f.write(f"exhaustiveness = {EXHAUSTIVENESS}\n")
        f.write(f"num_modes = {NUM_MODES}\n")
        f.write(f"energy_range = {ENERGY_RANGE}\n\n")

        f.write(f"out = {cfg['out']}\n")

    print(f"\nConfiguración creada: {conf.name}")
    print(f"Salida prevista      : {cfg['out']}")

print("\n" + "=" * 78)
print("Preparación LOCAL completada.")
print("=" * 78)