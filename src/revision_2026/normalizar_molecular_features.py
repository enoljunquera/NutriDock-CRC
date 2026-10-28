#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare a molecular feature table with one row per CID–receptor pair.

Usage: python normalizar_molecular_features.py input.csv output.csv
"""
import argparse
from pathlib import Path
import pandas as pd

PROT_COLS = [
    'Affinity1_rigid', 'Affinity2_flex',
    'c0_core_p95', 'c1_core_p95', 'c2_core_max', 'c2_core_p95',
    'b4_core_mean', 'b7_core_mean', 'c2_shell1_max', 'c2_shell1_p95',
    'b4_shell1_mean', 'b7_shell1_mean', 'c2_shell2_max',
    'c2_shell2_p95', 'b4_shell2_mean', 'b7_shell2_mean'
]
DESCRIPTOR_COLS = ['smiles', 'SMILES_MW', 'SMILES_TPSA',
                   'SMILES_logP', 'SMILES_NumRotBonds', 'nombre', 'Tipo']
ORDER = ['CYP1A1_4I8V', 'CYP1A2_2HI4', 'CYP2E1_3KOH', 'CYP3A4_1TQN']


def canonicalize(df: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
    source = df.copy()
    required = ['CID', 'Proteina'] + PROT_COLS + DESCRIPTOR_COLS
    missing = sorted(set(required) - set(source.columns))
    if missing:
        raise ValueError(f'Required columns are missing: {missing}')
    source['CID'] = pd.to_numeric(source['CID'], errors='raise').astype(int)
    source['Proteina'] = source['Proteina'].astype(str).str.strip()
    invalid = set(source['Proteina']) - set(ORDER)
    if invalid:
        raise ValueError(f'Unexpected receptor identifiers: {invalid}')
    source[PROT_COLS] = source[PROT_COLS].apply(pd.to_numeric, errors='raise')
    if source[PROT_COLS].isna().any().any():
        raise ValueError('Missing interaction values')
    groups = []
    for (_, _), group in source.groupby(['CID', 'Proteina'], sort=False):
        for column in DESCRIPTOR_COLS:
            if group[column].nunique(dropna=False) > 1:
                raise ValueError(f'Inconsistent descriptor: {column}')
        row = group.iloc[0].copy()
        if len(group) > 1:
            row.loc[PROT_COLS] = group[PROT_COLS].mean().to_numpy()
        groups.append(row)
    output = pd.DataFrame(groups, columns=source.columns)
    output['_order'] = pd.Categorical(output['Proteina'], categories=ORDER, ordered=True)
    output = (output.sort_values(['_order', 'CID'], kind='stable')
              .drop(columns='_order').reset_index(drop=True))
    if output.duplicated(['CID', 'Proteina']).any():
        raise ValueError('Repeated CID-receptor keys')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    prepared = canonicalize(pd.read_csv(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(args.output, index=False)
    print(f'Prepared {len(prepared)} CID-receptor rows')


if __name__ == '__main__':
    main()
