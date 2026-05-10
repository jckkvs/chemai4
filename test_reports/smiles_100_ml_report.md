# EDA Report: smiles_100_ml.csv

## Overview
- Shape: 100 rows × 5 columns
- Data type: smiles
- Target column: Not found

## Column Statistics
### Compound_Name (object)
- Non-null: 100 / 100
- Unique values: 100
### SMILES (object)
- Non-null: 100 / 100
- Unique values: 13
### logS (float64)
- Non-null: 100 / 100
- Mean: -2.7390, Std: 2.9778
- Range: [-13.7690, -0.2580]
- Unique values: 99
### MolecularWeight (float64)
- Non-null: 100 / 100
- Mean: 288.0723, Std: 118.9958
- Range: [102.2088, 494.7548]
- Unique values: 100
### NumAtoms (int64)
- Non-null: 100 / 100
- Mean: 27.2400, Std: 13.5759
- Range: [5.0000, 49.0000]
- Unique values: 38

## Correlation Matrix (numeric columns)
```
                     logS  MolecularWeight  NumAtoms
logS             1.000000        -0.119600  0.002711
MolecularWeight -0.119600         1.000000  0.158348
NumAtoms         0.002711         0.158348  1.000000
```