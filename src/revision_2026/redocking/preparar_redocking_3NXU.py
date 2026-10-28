from pathlib import Path

ROOT = Path(__file__).resolve().parent
FOLDER = ROOT / "CYP3A4_3NXU_RIT"

RECEPTOR = FOLDER / "CYP3A4_3NXU_chainA_clean_HEM.pdbqt"
LIGAND = FOLDER / "redock_RIT_3NXU.pdbqt"
CRYSTAL = FOLDER / "crystal_RIT_A_600.pdb"

CONF = FOLDER / "conf_redock_rigid.txt"
OUT = "redock_rigid_RIT_3NXU.pdbqt"

PADDING = 10.0  # Å por cada lado
MIN_SIZE = 24.0

EXHAUSTIVENESS = 24
NUM_MODES = 9
ENERGY_RANGE = 3


def read_xyz(path, heavy_only=False):
    atoms = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:

            if not line.startswith(("ATOM", "HETATM")):
                continue

            atom_name = line[12:16].strip()

            # Para definir la caja usamos solo átomos pesados.
            if heavy_only and atom_name.upper().startswith("H"):
                continue

            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue

            atoms.append((x, y, z))

    return atoms


def inside_box(atom, center, size):
    return all(
        c - s/2 <= coord <= c + s/2
        for coord, c, s in zip(atom, center, size)
    )


if not RECEPTOR.is_file():
    raise FileNotFoundError(RECEPTOR)

if not LIGAND.is_file():
    raise FileNotFoundError(LIGAND)

if not CRYSTAL.is_file():
    raise FileNotFoundError(CRYSTAL)


# ---------------------------------------------------------
# Caja a partir del RIT cristalográfico
# ---------------------------------------------------------

crystal_atoms = read_xyz(CRYSTAL, heavy_only=True)

if not crystal_atoms:
    raise RuntimeError("No se pudieron leer átomos de RIT cristalográfico.")

xs = [a[0] for a in crystal_atoms]
ys = [a[1] for a in crystal_atoms]
zs = [a[2] for a in crystal_atoms]

mins = (min(xs), min(ys), min(zs))
maxs = (max(xs), max(ys), max(zs))

center = tuple(
    (lo + hi) / 2
    for lo, hi in zip(mins, maxs)
)

span = tuple(
    hi - lo
    for lo, hi in zip(mins, maxs)
)

size = tuple(
    max(s + 2 * PADDING, MIN_SIZE)
    for s in span
)


# ---------------------------------------------------------
# Auditar el PDBQT completo de RIT
# ---------------------------------------------------------

ligand_atoms = read_xyz(LIGAND)

outside = [
    atom for atom in ligand_atoms
    if not inside_box(atom, center, size)
]


print("=" * 72)
print("CYP3A4 / 3NXU / RIT | caja de redocking")
print("=" * 72)

print(f"\nÁtomos pesados RIT cristalográfico: {len(crystal_atoms)}")
print(f"Átomos PDBQT RIT                : {len(ligand_atoms)}")

print(
    f"\nCentro:\n"
    f"  X = {center[0]:.3f}\n"
    f"  Y = {center[1]:.3f}\n"
    f"  Z = {center[2]:.3f}"
)

print(
    f"\nTamaño:\n"
    f"  X = {size[0]:.3f}\n"
    f"  Y = {size[1]:.3f}\n"
    f"  Z = {size[2]:.3f}"
)

if outside:
    print(f"\n[ERROR] {len(outside)} átomos de RIT quedan fuera.")
    raise SystemExit(1)

print("\nAUDITORÍA: todo RIT queda dentro de la caja.")


# ---------------------------------------------------------
# Crear configuración Vina
# ---------------------------------------------------------

with open(CONF, "w", encoding="utf-8") as f:

    f.write(f"receptor = {RECEPTOR.name}\n")
    f.write(f"ligand = {LIGAND.name}\n\n")

    f.write(f"center_x = {center[0]:.3f}\n")
    f.write(f"center_y = {center[1]:.3f}\n")
    f.write(f"center_z = {center[2]:.3f}\n\n")

    f.write(f"size_x = {size[0]:.3f}\n")
    f.write(f"size_y = {size[1]:.3f}\n")
    f.write(f"size_z = {size[2]:.3f}\n\n")

    f.write(f"exhaustiveness = {EXHAUSTIVENESS}\n")
    f.write(f"num_modes = {NUM_MODES}\n")
    f.write(f"energy_range = {ENERGY_RANGE}\n\n")

    f.write(f"out = {OUT}\n")


print(f"\nConfiguración creada: {CONF.name}")
print(f"Salida prevista      : {OUT}")

print("\n" + "=" * 72)
print("Preparación completada.")
print("=" * 72)