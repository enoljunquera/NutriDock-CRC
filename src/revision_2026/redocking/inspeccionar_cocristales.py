from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent

SYSTEMS = {
    "CYP1A1_4I8V_BHF": "BHF",
    "CYP1A2_2HI4_BHF": "BHF",
    "CYP2E1_3KOH_OIO": "OIO",
}


def find_original_pdb(folder):
    pdbs = [
        p for p in folder.glob("*.pdb")
        if "clean" not in p.name.lower()
    ]

    if len(pdbs) == 1:
        return pdbs[0]

    if len(pdbs) == 0:
        return None

    print(f"  [AVISO] Varios PDB encontrados:")
    for p in pdbs:
        print(f"          - {p.name}")
    return pdbs[0]


def inspect_ligand(pdb_file, ligand_name):
    copies = defaultdict(list)

    with open(pdb_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("HETATM"):
                continue

            resname = line[17:20].strip()

            if resname != ligand_name:
                continue

            chain = line[21].strip() or "-"
            resseq = line[22:26].strip()
            icode = line[26].strip() or "-"

            copies[(chain, resseq, icode)].append(line)

    return copies


print("=" * 72)
print("NutriDock-CRC | inspección de ligandos co-cristalizados")
print("=" * 72)

for folder_name, ligand_name in SYSTEMS.items():

    print(f"\n{folder_name}")
    print("-" * len(folder_name))

    folder = ROOT / folder_name

    if not folder.is_dir():
        print("[ERROR] No existe la carpeta.")
        continue

    pdb_file = find_original_pdb(folder)

    if pdb_file is None:
        print("[ERROR] No se encontró ningún PDB original.")
        continue

    clean_files = list(folder.glob("*clean*.pdbqt"))

    print(f"PDB original: {pdb_file.name}")

    if clean_files:
        print("Receptor clean:")
        for f in clean_files:
            print(f"  - {f.name}")
    else:
        print("[AVISO] No se encontró receptor clean PDBQT.")

    copies = inspect_ligand(pdb_file, ligand_name)

    print(f"Ligando esperado: {ligand_name}")
    print(f"Copias encontradas: {len(copies)}")

    for (chain, resseq, icode), atoms in copies.items():
        print(
            f"  Cadena={chain:>2} | "
            f"Residuo={resseq:>4} | "
            f"iCode={icode} | "
            f"Átomos={len(atoms)}"
        )

print("\n" + "=" * 72)
print("Inspección terminada.")
print("=" * 72)