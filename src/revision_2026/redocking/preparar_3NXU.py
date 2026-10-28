from pathlib import Path

ROOT = Path(__file__).resolve().parent
FOLDER = ROOT / "CYP3A4_3NXU_RIT"

INPUT = FOLDER / "3NXU.pdb"
RECEPTOR_OUT = FOLDER / "CYP3A4_3NXU_chainA_clean.pdb"
LIGAND_OUT = FOLDER / "crystal_RIT_A_600.pdb"

if not INPUT.is_file():
    raise FileNotFoundError(INPUT)

with open(INPUT, "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

receptor_lines = []
ligand_lines = []

n_protein = 0
n_hem = 0
n_rit = 0

for line in lines:

    if not line.startswith(("ATOM", "HETATM")):
        continue

    record = line[:6].strip()
    resname = line[17:20].strip()
    chain = line[21].strip()

    # ----------------------------
    # Ligando cristalográfico RIT
    # ----------------------------
    if (
        record == "HETATM"
        and resname == "RIT"
        and chain == "A"
    ):
        ligand_lines.append(line)
        n_rit += 1
        continue

    # ----------------------------
    # Receptor: SOLO cadena A
    # ----------------------------
    if chain != "A":
        continue

    # Proteína estándar
    if record == "ATOM":
        receptor_lines.append(line)
        n_protein += 1
        continue

    # Conservar HEM
    if record == "HETATM" and resname == "HEM":
        receptor_lines.append(line)
        n_hem += 1
        continue

    # El resto de heteroátomos:
    # RIT, HOH, DMS, etc. se eliminan


if not ligand_lines:
    raise RuntimeError("No se encontró RIT cadena A.")

if not receptor_lines:
    raise RuntimeError("No se generó receptor.")

with open(RECEPTOR_OUT, "w", encoding="utf-8") as out:
    out.write("REMARK CYP3A4 3NXU chain A; RIT/waters/other heteroatoms removed; HEM retained\n")
    for line in receptor_lines:
        out.write(line)
    out.write("END\n")

with open(LIGAND_OUT, "w", encoding="utf-8") as out:
    out.write("REMARK Native co-crystallized RIT, chain A residue 600\n")
    for line in ligand_lines:
        out.write(line)
    out.write("END\n")

print("=" * 72)
print("CYP3A4 / 3NXU preparation")
print("=" * 72)
print()
print(f"Protein atoms retained : {n_protein}")
print(f"HEM atoms retained     : {n_hem}")
print(f"RIT atoms extracted    : {n_rit}")
print()
print(f"Receptor PDB : {RECEPTOR_OUT.name}")
print(f"Ligand PDB   : {LIGAND_OUT.name}")
print()
print("Preparation completed.")