# EDA Report: smiles_25_quick.csv

## Overview
- Shape: 25 rows × 4 columns
- Data type: smiles
- Target column: Class

## Column Statistics
### Compound_Name (object)
- Non-null: 25 / 25
- Unique values: 25
### SMILES (object)
- Non-null: 25 / 25
- Unique values: 13
### logS (float64)
- Non-null: 25 / 25
- Mean: -2.9322, Std: 3.0861
- Range: [-12.6020, -0.0530]
- Unique values: 25
### Class (int64)
- Non-null: 25 / 25
- Mean: 0.4400, Std: 0.5066
- Range: [0.0000, 1.0000]
- Unique values: 2

## Correlation Matrix (numeric columns)
```
           logS     Class
logS   1.000000 -0.110954
Class -0.110954  1.000000
```