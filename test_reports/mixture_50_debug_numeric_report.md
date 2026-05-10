# EDA Report: mixture_50_debug_numeric.csv

## Overview
- Shape: 50 rows × 17 columns
- Data type: mixture
- Target column: Not found

## Column Statistics
### Sample_ID (object)
- Non-null: 50 / 50
- Unique values: 50
### Compound_1_Name (object)
- Non-null: 50 / 50
- Unique values: 8
### Compound_2_Name (object)
- Non-null: 50 / 50
- Unique values: 8
### Compound_3_Name (object)
- Non-null: 50 / 50
- Unique values: 8
### Compound_1_SMILES (object)
- Non-null: 50 / 50
- Unique values: 8
### Compound_1_WT% (float64)
- Non-null: 50 / 50
- Mean: 31.9840, Std: 13.8997
- Range: [10.4800, 61.8900]
- Unique values: 50
### Compound_2_SMILES (object)
- Non-null: 50 / 50
- Unique values: 8
### Compound_2_WT% (float64)
- Non-null: 50 / 50
- Mean: 35.5928, Std: 13.3451
- Range: [9.2300, 68.0700]
- Unique values: 50
### Compound_3_SMILES (object)
- Non-null: 50 / 50
- Unique values: 8
### Compound_3_WT% (float64)
- Non-null: 50 / 50
- Mean: 32.4236, Std: 15.2287
- Range: [8.6400, 68.9900]
- Unique values: 50
### Temperature_C (float64)
- Non-null: 50 / 50
- Mean: 49.6388, Std: 16.3133
- Range: [20.1200, 78.3600]
- Unique values: 50
### Humidity_pct (float64)
- Non-null: 50 / 50
- Mean: 65.2588, Std: 17.1351
- Range: [33.4500, 88.5300]
- Unique values: 50
### Pressure_atm (float64)
- Non-null: 50 / 50
- Mean: 1.3964, Std: 0.3631
- Range: [0.8080, 1.9970]
- Unique values: 49
### pH (float64)
- Non-null: 50 / 50
- Mean: 7.1754, Std: 1.6580
- Range: [4.0000, 9.9900]
- Unique values: 50
### StirringSpeed_rpm (float64)
- Non-null: 50 / 50
- Mean: 544.2160, Std: 268.0977
- Range: [106.7000, 986.0000]
- Unique values: 49
### ReactionTime_h (float64)
- Non-null: 50 / 50
- Mean: 12.6208, Std: 5.9785
- Range: [0.7100, 23.2600]
- Unique values: 50
### Boiling_Point_C(Target) (float64)
- Non-null: 50 / 50
- Mean: 2.9070, Std: 1.9217
- Range: [-2.1070, 7.2340]
- Unique values: 49

## Correlation Matrix (numeric columns)
```
                         Compound_1_WT%  Compound_2_WT%  Compound_3_WT%  Temperature_C  Humidity_pct  Pressure_atm        pH  StirringSpeed_rpm  ReactionTime_h  Boiling_Point_C(Target)
Compound_1_WT%                 1.000000       -0.375716       -0.583450      -0.195541      0.080354      0.057297  0.242970          -0.032376       -0.085208                -0.182650
Compound_2_WT%                -0.375716        1.000000       -0.533436       0.082529     -0.060719     -0.093052  0.040902          -0.003417       -0.048506                 0.241873
Compound_3_WT%                -0.583450       -0.533436        1.000000       0.106157     -0.020137      0.029261 -0.257635           0.032578        0.120227                -0.045208
Temperature_C                 -0.195541        0.082529        0.106157       1.000000      0.193828     -0.087855  0.096495           0.175664        0.008923                 0.593605
Humidity_pct                   0.080354       -0.060719       -0.020137       0.193828      1.000000     -0.192290  0.074180           0.033404        0.332883                 0.201702
Pressure_atm                   0.057297       -0.093052        0.029261      -0.087855     -0.192290      1.000000 -0.042378           0.111793        0.063500                 0.073806
pH                             0.242970        0.040902       -0.257635       0.096495      0.074180     -0.042378  1.000000           0.079528       -0.090879                -0.071366
StirringSpeed_rpm             -0.032376       -0.003417        0.032578       0.175664      0.033404      0.111793  0.079528           1.000000       -0.070153                 0.235447
ReactionTime_h                -0.085208       -0.048506        0.120227       0.008923      0.332883      0.063500 -0.090879          -0.070153        1.000000                 0.134117
Boiling_Point_C(Target)       -0.182650        0.241873       -0.045208       0.593605      0.201702      0.073806 -0.071366           0.235447        0.134117                 1.000000
```