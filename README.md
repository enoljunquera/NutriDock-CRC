# NutriDock-CRC

NutriDock-CRC is a subject-level cheminformatics framework designed to study whether individualized dietary exposure, together with molecular and protein-level features derived from docking and descriptor aggregation, can support colorectal alteration screening. The repository supports both a three-class setting (`Normal`, `Polyp`, `Neoplasia`) and a screening-oriented binary setting (`Normal` vs `Altered`, where `Altered` groups `Polyp` and `Neoplasia`).

The current processed cohort contains 111 subjects. In the associated study, the class distribution is 37 `Normal`, 66 `Polyp`, and 8 `Neoplasia`, which makes the three-class task substantially more unstable than the binary screening formulation.

## Overview

The repository implements a workflow with four main stages:

1. Construction of an integrated subject-level feature table from dietary exposure and molecular data.
2. Three-class predictive modelling with tabular neural and boosting baselines.
3. Binary screening experiments in the `Normal` versus `Altered` setting.
4. Auxiliary analyses, including Neoplasia-focused evaluation, ablations, experiment summarization, and exploratory learning-curve projection.

## Repository contents

### Input data

- `base_dieta.csv`  
  Subject-level dietary exposure table. It contains the dietary variables used as the basis for the subject-level exposure representation.

- `molecular_features.csv`  
  Molecular feature table containing ligand- and protein-level descriptors derived from SMILES, docking, and interaction summarization.

- `combined_features.csv`  
  Final integrated subject-level training table generated from the previous sources. In the current setup it contains 111 rows and the combined feature representation used in downstream modelling.

### Main scripts

- `combine_features.py`  
  Builds the integrated feature table from `base_dieta.csv` and `molecular_features.csv`. It computes the subject-level representations corresponding to dietary composition, ligand-level aggregation, and protein-specific aggregation, with configurable weighting and aggregation options.

- `train_MLP.py`  
  Baseline three-class tabular MLP training with stratified cross-validation and out-of-fold predictions. It also includes an optional graph-augmented mode that can incorporate graph embeddings through an additional GNN branch if graph resources are provided.

- `exp1_repeated_mlp.py`  
  Repeated stratified evaluation for the three-class MLP model. It uses internal early stopping on training-side data and stores fold-level metrics and OOF predictions.

- `exp3_boosting_repeated.py`  
  Repeated three-class boosting baseline. It attempts to use CatBoost or XGBoost if available, and otherwise falls back to scikit-learn’s histogram-based gradient boosting classifier.

- `neoplasia_metrics.py`  
  Computes Neoplasia-focused one-vs-rest metrics from OOF predictions, including per-split statistics and threshold sweeps over the predicted Neoplasia probability.

- `exp_bin_normal_vs_abnormal.py`  
  Binary screening experiments in the `Normal` versus `Altered` setting. Supports threshold selection with either a Youden-type criterion or a fixed-specificity target.

- `exp_bin_normal_vs_abnormal_nested.py`  
  Nested binary MLP refinement with a restricted hyperparameter search around the reference binary architecture.

- `learning_curve_projection.py`  
  Exploratory projection of binary screening performance under cohort expansion, using repeated stratified subsampling and cross-validation without recomputing features.

- `option4_ablation_pipeline.py`  
  Ablation pipeline for comparing alternative subject-level feature-construction strategies, including different weighting and aggregation choices.

- `summarize_experiments.py`  
  Aggregates metrics and OOF outputs across experiments and generates consolidated summaries, including bootstrap confidence intervals for mean metrics when applicable.

## Methodological summary

The framework combines three complementary feature blocks at subject level:

- a **Diet** block representing normalized dietary chemical contributions together with total intake,
- a **MolAgg** block representing exposure-weighted ligand-level molecular descriptors,
- and a **ProtAgg** block representing exposure-weighted protein-specific interaction summaries.

These blocks are concatenated into a final combined representation for downstream modelling. In the associated study, the framework links estimated annual exposure to 37 dietary chemicals with docking-derived molecular summaries built from 71 ligands and four cytochrome P450 enzymes: `CYP1A1`, `CYP1A2`, `CYP2E1`, and `CYP3A4`.

## Typical workflow

A common end-to-end workflow is:

1. Build the integrated feature table.
2. Run the baseline three-class experiment.
3. Run repeated three-class experiments.
4. Compute Neoplasia-specific analyses.
5. Run binary screening experiments.
6. Summarize all outputs.

## Quickstart

### 1. Build subject-level features

```bash
python combine_features.py
```

Depending on the local version and arguments used, the script may also be executed with explicit input and output paths. The usual pattern is:

```bash
python combine_features.py --base base_dieta.csv --molecular molecular_features.csv --out combined_features.csv
```

This step generates the final integrated subject-level table used by the downstream models.

### 2. Train the baseline three-class MLP

```bash
python train_MLP.py --csv combined_features.csv --out_prefix tab_mlp
```

This produces out-of-fold predictions and summary metrics for the baseline three-class tabular MLP.

### 3. Run repeated three-class experiments

```bash
python exp1_repeated_mlp.py --csv combined_features.csv --repeats 20 --out_prefix exp1_mlp_R20
python exp3_boosting_repeated.py --csv combined_features.csv --repeats 20 --out_prefix exp3_boost_R20
```

These scripts generate repeated stratified evaluations for the MLP and boosting baselines, respectively.

### 4. Run Neoplasia-specific evaluation

```bash
python neoplasia_metrics.py --oof_csv exp1_mlp_R20_oof.csv --out_prefix neo_exp1_mlp
python neoplasia_metrics.py --oof_csv exp3_boost_R20_sk_hgb_oof.csv --out_prefix neo_exp3_hgb
```

This step derives Neoplasia-oriented statistics from the repeated OOF predictions.

### 5. Run binary screening experiments

```bash
python exp_bin_normal_vs_abnormal.py --csv combined_features.csv --model mlp --repeats 20 --thr_mode youden --out_prefix bin_R20

python exp_bin_normal_vs_abnormal.py --csv combined_features.csv --model hgb --repeats 20 --thr_mode youden --out_prefix bin_R20

python exp_bin_normal_vs_abnormal.py --csv combined_features.csv --model hgb --repeats 20 --thr_mode fixed_spec --target_spec 0.90 --out_prefix bin_R20
```

The binary formulation is intended for screening-oriented evaluation, where `Altered` merges `Polyp` and `Neoplasia`.

### 6. Summarize all experiments

```bash
python summarize_experiments.py --dir . --out_xlsx results_summary.xlsx --out_csv results_summary.csv
```

This script scans the experiment outputs and generates consolidated summary tables.

## Main outputs

Typical outputs generated by the repository include:

- `*_split_metrics.csv`: fold-level performance metrics,
- `*_oof.csv`: out-of-fold predictions and probabilities,
- `*_oof_probs.npy` and `*_oof_pred.npy`: NumPy outputs from the base MLP training,
- `results_summary.csv` and `results_summary.xlsx`: consolidated summaries across experiments.

## Current interpretation of results

The present repository supports both exploratory three-class modelling and a more practically relevant binary screening formulation. In the current cohort, the three-class setting is limited by the scarcity of `Neoplasia` cases, whereas the `Normal` versus `Altered` setting provides the clearest applied signal. The associated study therefore interprets the framework primarily as a prioritization tool for colorectal screening research rather than as a finished diagnostic system.

## Reproducibility notes

This repository is intended to provide the processed data products and scripts required to reproduce the reported modelling workflow. Because the study involves human dietary and clinical data, raw individual-level data may remain subject to ethical and legal restrictions, while processed and de-identified data products are the intended reproducible outputs.

## Requirements

The code relies mainly on:

- Python 3
- NumPy
- pandas
- scikit-learn
- PyTorch

Some optional paths may additionally use:

- torch-geometric, for graph-based experiments in `train_MLP.py`,
- CatBoost or XGBoost, for alternative boosting backends in `exp3_boosting_repeated.py`.


## Citation

If you use this repository, please cite the corresponding NutriDock-CRC study and the repository DOI when available.

