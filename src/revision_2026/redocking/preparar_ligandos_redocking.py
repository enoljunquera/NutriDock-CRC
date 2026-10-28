from pathlib import Path
import subprocess
import shutil
import sys

ROOT = Path(__file__).resolve().parent

SYSTEMS = {
    "CYP1A1_4I8V_BHF": {
        "input": "crystal_BHF_B_602.pdb",
        "output": "redock_BHF_4I8V.pdbqt",
    },
    "CYP1A2_2HI4_BHF": {
        "input": "crystal_BHF_A_800.pdb",
        "output": "redock_BHF_2HI4.pdbqt",
    },
    "CYP2E1_3KOH_OIO": {
        "input": "crystal_OIO_B_1.pdb",
        "output": "redock_OIO_3KOH.pdbqt",
    },
}

OBABEL = shutil.which("obabel")

if OBABEL is None:
    print("[ERROR] No encuentro 'obabel' en PATH.")
    print("Ejecuta este script desde el entorno donde tengas Open Babel disponible.")
    sys.exit(1)

print("=" * 72)
print("NutriDock-CRC | preparación de ligandos para redocking")
print("=" * 72)
print(f"Open Babel: {OBABEL}")

for folder_name, cfg in SYSTEMS.items():

    folder = ROOT / folder_name
    input_file = folder / cfg["input"]
    output_file = folder / cfg["output"]

    print("\n" + "-" * 72)
    print(folder_name)
    print("-" * 72)

    if not input_file.is_file():
        print(f"[ERROR] No existe: {input_file.name}")
        continue

    cmd = [
        OBABEL,
        str(input_file),
        "-O", str(output_file),
        "-h",
        "--partialcharge", "gasteiger",
    ]

    print(f"Entrada : {input_file.name}")
    print(f"Salida  : {output_file.name}")
    print("Sin --gen3d: se conservan las coordenadas cristalográficas.")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    if result.stdout.strip():
        print(result.stdout.strip())

    if result.stderr.strip():
        print(result.stderr.strip())

    if result.returncode != 0:
        print(f"[ERROR] Open Babel terminó con código {result.returncode}")
        continue

    if not output_file.is_file() or output_file.stat().st_size == 0:
        print("[ERROR] El PDBQT no se generó correctamente.")
        continue

    print(f"OK → {output_file.name} ({output_file.stat().st_size} bytes)")

print("\n" + "=" * 72)
print("Preparación terminada.")
print("=" * 72)