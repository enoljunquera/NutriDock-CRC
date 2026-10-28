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

The code is retained as research scripts and is not a standalone, containerized replication package. Some analyses require authorized access to the cohort-level dietary and diagnostic tables, as well as intermediate molecular structures.

## Molecular features

The released table uses one row per compound (PubChem CID) and receptor. When multiple source records are available for a CID–receptor pair, numerical protein-interaction descriptors are summarized by their arithmetic mean. The four receptor identifiers are `CYP1A1_4I8V`, `CYP1A2_2HI4`, `CYP3A4_1TQN` and `CYP2E1_3KOH`.

## Data and citation

The associated dataset is archived in Zenodo: https://doi.org/10.5281/zenodo.19555687. Use the DOI for the specific dataset version consulted when citing the molecular table.

Please verify licenses and permission requirements before redistributing data.


NutriDock-CRC: A reproducible cheminformatics framework for colorectal cancer risk prediction integrating molecular docking, CYP interactions, exposure data, and machine learning. Original source code released under the MIT License (https://opensource.org/license/mit/).
