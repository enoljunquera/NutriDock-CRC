import numpy as np
import pandas as pd
from pathlib import Path
import re


def make_masks(size=25, core=9, mid=17):
    assert size % 2 == 1, "size debería ser impar (25)"
    assert core % 2 == 1 and mid % 2 == 1, "core y mid deberían ser impares"
    assert core < mid < size

    center = size // 2

    def cube_slice(side):
        half = side // 2
        return slice(center - half, center + half + 1)

    core_sl = (cube_slice(core),) * 3
    mid_sl = (cube_slice(mid),) * 3

    mask_core = np.zeros((size, size, size), dtype=bool)
    mask_mid = np.zeros((size, size, size), dtype=bool)

    mask_core[core_sl] = True
    mask_mid[mid_sl] = True

    mask_shell1 = mask_mid & (~mask_core)
    mask_shell2 = (~mask_mid)

    return {"core": mask_core, "shell1": mask_shell1, "shell2": mask_shell2}


def summarize_voxel(
    voxel,
    masks,
    continuous_idx=(0, 1, 2, 3),
    boolean_idx=(4, 5, 6, 7),
    hi_perc=(90, 95),
):
    feats = {}
    for region, mask in masks.items():
        for c in continuous_idx:
            vals = voxel[..., c][mask]
            feats[f"c{c}_{region}_max"] = float(np.max(vals))
            for p in hi_perc:
                feats[f"c{c}_{region}_p{p}"] = float(np.percentile(vals, p))

        for c in boolean_idx:
            vals = voxel[..., c][mask]
            feats[f"b{c}_{region}_mean"] = float(np.mean(vals))
    return feats


def infer_ids_from_filename(fname: str):
    base = Path(fname).stem

    cid = None
    m = re.search(r"CID(\d+)", base)
    if m:
        cid = m.group(1)

    target = re.sub(r"_25x25x25x8$", "", base)

    # chem_id SIN el prefijo "CID"
    chem_id = cid if cid else base

    return chem_id, target



def build_features_from_npy(npy_path: Path, masks):
    voxel = np.load(npy_path)

    if voxel.shape != (25, 25, 25, 8):
        raise ValueError(f"Shape inesperado: {voxel.shape}")

    chem_id, target = infer_ids_from_filename(npy_path.name)
    feats = summarize_voxel(voxel, masks)

    row = {
        "chem_id": chem_id,
        "target": target,
        "source_file": npy_path.name,
    }
    row.update(feats)
    return row


def main():
    # Carpeta donde está ESTE script
    script_dir = Path(__file__).resolve().parent

    npy_files = sorted(script_dir.glob("*.npy"))
    if not npy_files:
        print(f"❌ No encontré archivos .npy en: {script_dir}")
        return

    masks = make_masks(size=25, core=9, mid=17)

    rows = []
    skipped = []
    for npy_path in npy_files:
        try:
            rows.append(build_features_from_npy(npy_path, masks))
        except Exception as e:
            skipped.append((npy_path.name, str(e)))

    if not rows:
        print("❌ No se pudo procesar ningún .npy (todos fallaron).")
        print("Detalles:", skipped[:10])
        return

    df = pd.DataFrame(rows)

    out_path = script_dir / "chem_voxel_features.parquet"
    df.to_parquet(out_path, index=False)

    print(f"✅ OK: {len(rows)} archivos procesados")
    print(f"📦 Guardado: {out_path}")

    if skipped:
        print(f"⚠️ Saltados: {len(skipped)} archivos por error/shape.")
        for name, err in skipped[:10]:
            print(f"  - {name}: {err}")
        if len(skipped) > 10:
            print("  (mostrando solo los 10 primeros)")


if __name__ == "__main__":
    main()
