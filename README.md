# NutriDock-CRC

Computational framework for subject-level colorectal alteration modeling using estimated dietary exposure and docking-derived molecular interaction profiles.

## Study overview

The cohort comprises 111 participants (37 Normal, 66 Polyp, 8 Neoplasia). The curated ligand library comprises 63 compounds; 37 compounds have quantitative dietary exposure measurements and enter the subject-level prediction models. The molecular feature table contains 173 unique CID–protein pairs for CYP1A1, CYP1A2, CYP2E1 and CYP3A4.

Diet, MolAgg and ProtAgg are assessed separately and in the Combined representation. The three-class and binary Normal-versus-Altered analyses are exploratory. The Altered category is predominantly Polyp; the results do not establish CRC screening or clinical diagnostic performance. An independent clinical cohort is needed to assess generalizability.

## Repository contents

- `src/revision_2026/`: model evaluation, food-level baseline, docking and redocking analysis scripts.
- `data/molecular_features.csv`: molecular interaction and ligand descriptor table.
- `results/revision_2026/`: non-subject-level evaluation outputs, learning curves, statistical comparisons and redocking summaries.
- `DATA_AVAILABILITY.md`: data access and privacy considerations.


The code is provided as research scripts rather than as a standalone, containerized replication package. The pseudonymized subject-level feature matrices and diagnostic labels used in the reported predictive analyses are available in the associated Zenodo dataset (version 2). Some analyses additionally require intermediate docking outputs or molecular structures not included in the public release.

## Molecular features

The released table uses one row per compound (PubChem CID) and receptor. When multiple source records are available for a CID–receptor pair, numerical protein-interaction descriptors are summarized by their arithmetic mean. The four receptor identifiers are `CYP1A1_4I8V`, `CYP1A2_2HI4`, `CYP3A4_1TQN` and `CYP2E1_3KOH`.


## Data and citation

The processed datasets supporting the reported predictive analyses, including dietary exposure matrices, molecular descriptors, subject-level feature matrices, diagnostic labels, and evaluation results, are archived in Zenodo (version 2):

https://zenodo.org/records/23728824

Please cite the version-specific DOI displayed on the Zenodo record.

The source code is released under the MIT License. Data reuse and redistribution must comply with the applicable dataset license, research ethics approvals, and privacy requirements.
