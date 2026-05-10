# EDA Report: mixture_30_simple.csv

## Overview
- Shape: 30 rows × 13 columns
- Data type: mixture
- Target column: Target_Property

## Column Statistics
### Notes (object)
- Non-null: 30 / 30
- Unique values: 1
### Compound_1_SMILES (object)
- Non-null: 30 / 30
- Unique values: 7
### Compound_1_Name (object)
- Non-null: 30 / 30
- Unique values: 7
### Compound_1_WT% (float64)
- Non-null: 30 / 30
- Mean: 36.3397, Std: 12.6284
- Range: [12.9800, 59.4400]
- Unique values: 29
### Compound_2_SMILES (object)
- Non-null: 30 / 30
- Unique values: 7
### Compound_2_Name (object)
- Non-null: 30 / 30
- Unique values: 7
### Compound_2_WT% (float64)
- Non-null: 30 / 30
- Mean: 32.6853, Std: 12.9892
- Range: [10.2800, 71.6500]
- Unique values: 30
### Compound_3_SMILES (object)
- Non-null: 30 / 30
- Unique values: 8
### Compound_3_Name (object)
- Non-null: 30 / 30
- Unique values: 8
### Compound_3_WT% (float64)
- Non-null: 30 / 30
- Mean: 30.9757, Std: 14.5030
- Range: [10.7700, 68.4900]
- Unique values: 30
### Target_Property (float64)
- Non-null: 30 / 30
- Mean: 2.4701, Std: 2.5880
- Range: [-3.5080, 6.9360]
- Unique values: 30
### Sample_ID (object)
- Non-null: 30 / 30
- Unique values: 30
### Total_WT% (float64)
- Non-null: 30 / 30
- Mean: 100.0007, Std: 0.0058
- Range: [99.9900, 100.0100]
- Unique values: 3

## Correlation Matrix (numeric columns)
```
                 Compound_1_WT%  Compound_2_WT%  Compound_3_WT%  Target_Property  Total_WT%
Compound_1_WT%         1.000000       -0.359211       -0.549080        -0.084633  -0.128873
Compound_2_WT%        -0.359211        1.000000       -0.582752         0.316254   0.233113
Compound_3_WT%        -0.549080       -0.582752        1.000000        -0.209492  -0.096164
Target_Property       -0.084633        0.316254       -0.209492         1.000000   0.148336
Total_WT%             -0.128873        0.233113       -0.096164         0.148336   1.000000
```