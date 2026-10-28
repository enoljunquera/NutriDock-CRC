import pandas as pd
from pathlib import Path

# ============================================================
# ARCHIVOS DE ENTRADA
# ============================================================

BASE_DIETA = Path("Base dieta_.xlsx")
DESCRIPCION = Path("Descripción_variables base dieta_CID_v3_.xlsx")
DIAGNOSTICO = Path("Diagnóstico individuos_.xlsx")

SALIDA = Path("dataset_baseline_alimentos.csv")


# ============================================================
# 1. CARGAR LOS TRES ARCHIVOS
# ============================================================

dieta = pd.read_excel(
    BASE_DIETA,
    sheet_name="base combinada MIXED+MTD"
)

descripcion = pd.read_excel(
    DESCRIPCION,
    sheet_name="final"
)

diagnostico = pd.read_excel(
    DIAGNOSTICO,
    sheet_name="Diagnostico"
)


# ============================================================
# 2. IDENTIFICAR AUTOMÁTICAMENTE LAS VARIABLES DE ALIMENTOS
# ============================================================

variables_alimentos = (
    descripcion.loc[
        descripcion["Tipo"].eq("Alimento"),
        "Variable"
    ]
    .dropna()
    .astype(str)
    .tolist()
)

print(f"Variables clasificadas como Alimento: {len(variables_alimentos)}")

# En estos archivos deberían ser 307
assert len(variables_alimentos) == 307, (
    f"Esperaba 307 variables de alimentos, "
    f"pero encontré {len(variables_alimentos)}."
)


# ============================================================
# 3. COMPROBAR QUE TODAS EXISTEN EN LA BASE DE DIETA
# ============================================================

faltantes = [
    variable
    for variable in variables_alimentos
    if variable not in dieta.columns
]

if faltantes:
    raise ValueError(
        f"Hay variables clasificadas como Alimento "
        f"que no aparecen en Base dieta: {faltantes}"
    )


# ============================================================
# 4. QUEDARNOS SOLO CON ID + ALIMENTOS
# ============================================================

dataset = dieta[
    ["Codigo_dieta"] + variables_alimentos
].copy()


# ============================================================
# 5. PREPARAR DIAGNÓSTICOS
# ============================================================

columna_diagnostico = (
    "Descripción: Grupo de estudio "
    "(pólipo intestinal, neoplasia intestinal o individuo normal sin lesion)"
)

diag = diagnostico[
    ["Codigo_individuo", columna_diagnostico]
].copy()

diag = diag.rename(
    columns={
        "Codigo_individuo": "Codigo_dieta",
        columna_diagnostico: "Diagnostico"
    }
)


# ============================================================
# 6. UNIR DIETA Y DIAGNÓSTICO
# ============================================================

dataset = dataset.merge(
    diag,
    on="Codigo_dieta",
    how="inner",
    validate="one_to_one"
)


# ============================================================
# 7. CREAR ETIQUETAS PARA LOS DOS PROBLEMAS DEL ARTÍCULO
# ============================================================

# Etiqueta multiclase:
# 0 = Normal
# 1 = Pólipo
# 2 = Neoplasia

mapa_3clases = {
    "Normal": 0,
    "Pólipo": 1,
    "Neoplasia": 2
}

dataset["y_3class"] = dataset["Diagnostico"].map(mapa_3clases)


# Etiqueta binaria:
# 0 = Normal
# 1 = Altered (Pólipo + Neoplasia)

dataset["Grupo_binario"] = dataset["Diagnostico"].replace({
    "Pólipo": "Altered",
    "Neoplasia": "Altered"
})

dataset["y_binary"] = (
    dataset["Grupo_binario"] == "Altered"
).astype(int)


# ============================================================
# 8. CONTROLES DE CALIDAD
# ============================================================

print("\n--- CONTROL DEL DATASET ---")

print(f"Sujetos: {len(dataset)}")
print(f"Variables de alimentos: {len(variables_alimentos)}")

print("\nDiagnóstico:")
print(dataset["Diagnostico"].value_counts())

print("\nProblema binario:")
print(dataset["Grupo_binario"].value_counts())

print(
    "\nValores ausentes en variables de alimentos:",
    dataset[variables_alimentos].isna().sum().sum()
)

# Comprobaciones esperadas para NutriDock-CRC

assert len(dataset) == 111, (
    f"Esperaba 111 sujetos y hay {len(dataset)}."
)

assert dataset["Codigo_dieta"].nunique() == 111

assert dataset["Diagnostico"].value_counts().to_dict() == {
    "Pólipo": 66,
    "Normal": 37,
    "Neoplasia": 8
}

assert dataset["y_binary"].value_counts().to_dict() == {
    1: 74,
    0: 37
}

assert dataset[variables_alimentos].isna().sum().sum() == 0


# ============================================================
# 9. ORDENAR COLUMNAS
# ============================================================

dataset = dataset[
    ["Codigo_dieta"]
    + variables_alimentos
    + [
        "Diagnostico",
        "y_3class",
        "Grupo_binario",
        "y_binary"
    ]
]


# ============================================================
# 10. GUARDAR
# ============================================================

dataset.to_csv(
    SALIDA,
    index=False,
    encoding="utf-8-sig"
)

print(f"\nDataset guardado en: {SALIDA.resolve()}")
print(f"Dimensiones finales: {dataset.shape}")