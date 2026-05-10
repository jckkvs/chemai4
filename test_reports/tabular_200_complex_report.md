# EDA Report: tabular_200_complex.csv

## Overview
- Shape: 200 rows × 11 columns
- Data type: tabular
- Target column: Target

## Column Statistics
### Feature_1 (float64)
- Non-null: 195 / 200
- Mean: 30.1590, Std: 5.2615
- Range: [13.8447, 42.9915]
- Unique values: 195
### Feature_2 (float64)
- Non-null: 199 / 200
- Mean: 29.4483, Std: 4.6438
- Range: [18.7423, 44.7931]
- Unique values: 199
### Feature_3 (float64)
- Non-null: 195 / 200
- Mean: 30.4058, Std: 5.0392
- Range: [16.0594, 42.7993]
- Unique values: 195
### Feature_4 (float64)
- Non-null: 195 / 200
- Mean: 29.8496, Std: 5.2102
- Range: [16.0276, 47.8579]
- Unique values: 195
### Feature_5 (float64)
- Non-null: 200 / 200
- Mean: 29.4348, Std: 4.8159
- Range: [14.1647, 43.8330]
- Unique values: 200
### Feature_6 (float64)
- Non-null: 200 / 200
- Mean: 30.0738, Std: 4.8812
- Range: [16.0958, 42.7795]
- Unique values: 200
### Feature_7 (float64)
- Non-null: 200 / 200
- Mean: 30.0452, Std: 4.2966
- Range: [19.2028, 41.3552]
- Unique values: 200
### Feature_8 (float64)
- Non-null: 200 / 200
- Mean: 30.4137, Std: 4.7594
- Range: [15.3489, 41.8569]
- Unique values: 200
### Target (float64)
- Non-null: 200 / 200
- Mean: 77.9800, Std: 15.7048
- Range: [22.9793, 122.7606]
- Unique values: 200
### Sample_ID (object)
- Non-null: 200 / 200
- Unique values: 200
### Category (object)
- Non-null: 200 / 200
- Unique values: 3

## Correlation Matrix (numeric columns)
```
           Feature_1  Feature_2  Feature_3  Feature_4  Feature_5  Feature_6  Feature_7  Feature_8    Target
Feature_1   1.000000   0.000573  -0.004903  -0.030539  -0.113766  -0.132858  -0.012158  -0.058119  0.060213
Feature_2   0.000573   1.000000   0.018941   0.134418   0.045704  -0.045009   0.000532  -0.074902 -0.050223
Feature_3  -0.004903   0.018941   1.000000   0.011248   0.057406  -0.009004   0.077906  -0.050976  0.114763
Feature_4  -0.030539   0.134418   0.011248   1.000000   0.167302   0.029948  -0.036841   0.046473 -0.050124
Feature_5  -0.113766   0.045704   0.057406   0.167302   1.000000  -0.014902  -0.053664   0.142950 -0.191632
Feature_6  -0.132858  -0.045009  -0.009004   0.029948  -0.014902   1.000000  -0.054101   0.072356 -0.022571
Feature_7  -0.012158   0.000532   0.077906  -0.036841  -0.053664  -0.054101   1.000000  -0.044715 -0.017568
Feature_8  -0.058119  -0.074902  -0.050976   0.046473   0.142950   0.072356  -0.044715   1.000000 -0.039309
Target      0.060213  -0.050223   0.114763  -0.050124  -0.191632  -0.022571  -0.017568  -0.039309  1.000000
```