# EDA Report: mixture_100_ml.csv

## Overview
- Shape: 100 rows × 16 columns
- Data type: mixture
- Target column: Target_Property

## Column Statistics
### Sample_ID (object)
- Non-null: 100 / 100
- Unique values: 100
### Compound_1_Name (object)
- Non-null: 100 / 100
- Unique values: 100
### Compound_2_Name (object)
- Non-null: 100 / 100
- Unique values: 100
### Compound_3_Name (object)
- Non-null: 100 / 100
- Unique values: 100
### Compound_1_SMILES (object)
- Non-null: 100 / 100
- Unique values: 5
### Compound_2_SMILES (object)
- Non-null: 100 / 100
- Unique values: 5
### Compound_3_SMILES (object)
- Non-null: 100 / 100
- Unique values: 5
### Compound_1_WT% (float64)
- Non-null: 100 / 100
- Mean: 33.0723, Std: 8.7414
- Range: [20.1555, 49.4552]
- Unique values: 100
### Compound_2_WT% (float64)
- Non-null: 100 / 100
- Mean: 36.4578, Std: 8.6124
- Range: [20.4591, 49.8779]
- Unique values: 100
### Compound_3_WT% (float64)
- Non-null: 100 / 100
- Mean: 34.2534, Std: 8.4668
- Range: [20.4102, 49.8749]
- Unique values: 100
### Temperature_C (float64)
- Non-null: 100 / 100
- Mean: 85.2047, Std: 36.3123
- Range: [20.3374, 148.2134]
- Unique values: 100
### Humidity_pct (float64)
- Non-null: 100 / 100
- Mean: 62.1892, Std: 16.8737
- Range: [30.1627, 89.8353]
- Unique values: 100
### Pressure_atm (float64)
- Non-null: 100 / 100
- Mean: 1.2240, Std: 0.4156
- Range: [0.5183, 1.9965]
- Unique values: 100
### StirringSpeed_rpm (int64)
- Non-null: 100 / 100
- Mean: 306.5300, Std: 115.1626
- Range: [101.0000, 499.0000]
- Unique values: 92
### ReactionTime_h (float64)
- Non-null: 100 / 100
- Mean: 12.2897, Std: 6.3772
- Range: [1.2529, 23.6413]
- Unique values: 100
### Target_Property (float64)
- Non-null: 100 / 100
- Mean: 129.8998, Std: 47.9079
- Range: [52.6811, 199.6901]
- Unique values: 100

## Correlation Matrix (numeric columns)
```
                   Compound_1_WT%  Compound_2_WT%  Compound_3_WT%  Temperature_C  Humidity_pct  Pressure_atm  StirringSpeed_rpm  ReactionTime_h  Target_Property
Compound_1_WT%           1.000000       -0.103605       -0.066112      -0.153551      0.140341     -0.079276          -0.137019        0.150885         0.085181
Compound_2_WT%          -0.103605        1.000000        0.192476      -0.033675     -0.021289      0.215435           0.050212       -0.076921         0.154280
Compound_3_WT%          -0.066112        0.192476        1.000000       0.007649      0.069700      0.044625           0.061252       -0.054564         0.115854
Temperature_C           -0.153551       -0.033675        0.007649       1.000000     -0.062645     -0.088306           0.089701        0.112583         0.277603
Humidity_pct             0.140341       -0.021289        0.069700      -0.062645      1.000000      0.148159           0.036410        0.248351         0.082088
Pressure_atm            -0.079276        0.215435        0.044625      -0.088306      0.148159      1.000000           0.103866        0.031852        -0.065067
StirringSpeed_rpm       -0.137019        0.050212        0.061252       0.089701      0.036410      0.103866           1.000000        0.009539         0.254928
ReactionTime_h           0.150885       -0.076921       -0.054564       0.112583      0.248351      0.031852           0.009539        1.000000         0.115947
Target_Property          0.085181        0.154280        0.115854       0.277603      0.082088     -0.065067           0.254928        0.115947         1.000000
```