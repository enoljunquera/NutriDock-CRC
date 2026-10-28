from pathlib import Path
from collections import defaultdict
import re

ROOT = Path(__file__).resolve().parent

SYSTEMS = {
    "CYP1A1_4I8V_BHF": "BHF",
    "CYP1A2_2HI4_BHF": "BHF",
    "CYP2E1_3KOH_OIO": "OIO",
}


def find_original_pdb(folder):
    pdbs = [
        p for p in folder.glob("*.pdb")
        if not p.name.lower().startswith("crystal_")
        and "clean" not in p.name.lower()
    ]

    if len(pdbs) != 1:
        raise RuntimeError(
            f"{folder.name}: esperaba 1 PDB original y encontré {len(pdbs)}"
        )

    return pdbs[0]


def atom_serial(line):
    try:
        return int(line[6:11])
    except ValueError:
        return None


for folder_name, ligand in SYSTEMS.items():

    folder = ROOT / folder_name
    pdb_file = find_original_pdb(folder)

    print("\n" + "=" * 72)
    print(folder_name)
    print("=" * 72)
    print(f"PDB: {pdb_file.name}")

    with open(pdb_file, "r", encoding="utf-8", errors="ignore") as f:
        all_lines = f.readlines()

    copies = defaultdict(list)

    for line in all_lines:

        if not line.startswith("HETATM"):
            continue

        if line[17:20].strip() != ligand:
            continue

        chain = line[21].strip() or "-"
        resseq = line[22:26].strip()
        icode = line[26].strip() or "-"

        copies[(chain, resseq, icode)].append(line)

    if not copies:
        raise RuntimeError(
            f"{folder_name}: no se encontró ningún {ligand}"
        )

    for (chain, resseq, icode), ligand_lines in copies.items():

        output = folder / f"crystal_{ligand}_{chain}_{resseq}.pdb"

        selected_serials = {
            atom_serial(line)
            for line in ligand_lines
            if atom_serial(line) is not None
        }

        with open(output, "w", encoding="utf-8") as out:

            out.write(
                f"REMARK Native co-crystallized ligand {ligand}\n"
            )
            out.write(
                f"REMARK Chain {chain} residue {resseq}\n"
            )

            for line in ligand_lines:
                out.write(line)

            # Mantener únicamente CONECT pertenecientes
            # al ligando extraído.
            for line in all_lines:

                if not line.startswith("CONECT"):
                    continue

                serials = [
                    int(x)
                    for x in re.findall(r"\d+", line[6:])
                ]

                if not serials:
                    continue

                source = serials[0]

                if source not in selected_serials:
                    continue

                # Conservar únicamente conexiones entre átomos
                # pertenecientes al ligando extraído.
                neighbours = [
                    serial
                    for serial in serials[1:]
                    if serial in selected_serials
                ]

                if not neighbours:
                    continue

                conect_line = "CONECT" + f"{source:5d}"
                conect_line += "".join(f"{serial:5d}" for serial in neighbours)
                out.write(conect_line + "\n")

            out.write("END\n")

        print(
            f"OK  {ligand} | cadena {chain} | "
            f"residuo {resseq} | átomos {len(ligand_lines)} "
            f"→ {output.name}"
        )

print("\n" + "=" * 72)
print("Todas las referencias cristalográficas han sido extraídas.")
print("=" * 72)