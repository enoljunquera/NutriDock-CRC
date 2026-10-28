from pathlib import Path
import subprocess
import shutil
import sys

ROOT = Path(__file__).resolve().parent

SYSTEMS = [
    "CYP1A1_4I8V_BHF",
    "CYP1A2_2HI4_BHF",
    "CYP2E1_3KOH_OIO",
    "CYP3A4_3NXU_RIT",
]

VINA = shutil.which("vina.exe") or shutil.which("vina")

if VINA is None:
    print("[ERROR] No encuentro Vina en PATH.")
    sys.exit(1)

print("=" * 78)
print("NutriDock-CRC | redocking LOCAL de co-cristales")
print("=" * 78)
print(f"Vina: {VINA}")

failures = 0

for system in SYSTEMS:

    folder = ROOT / system
    conf = folder / "conf_redock_local.txt"
    log = folder / "redock_local.log"

    print("\n" + "=" * 78)
    print(system)
    print("=" * 78)

    if not conf.is_file():
        print(f"[ERROR] No existe {conf}")
        failures += 1
        continue

    cmd = [
        VINA,
        "--config",
        conf.name,
        "--log",
        log.name,
    ]

    print("Ejecutando:")
    print(" ".join(str(x) for x in cmd))
    print()

    result = subprocess.run(
        cmd,
        cwd=str(folder),
    )

    if result.returncode != 0:
        print(f"\n[ERROR] Vina terminó con código {result.returncode}")
        failures += 1
    else:
        print("\n[OK] Redocking local terminado.")
        print(f"Log: {log.name}")

print("\n" + "=" * 78)

if failures:
    print(f"Finalizado con {failures} error(es).")
    sys.exit(1)

print("Los cuatro redockings locales terminaron correctamente.")
print("=" * 78)