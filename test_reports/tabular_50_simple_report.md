# EDA Report: tabular_50_simple.csv

## Overview
- Shape: 50 rows × 9 columns
- Data type: tabular
- Target column: Target

## Column Statistics
### Feature_1 (float64)
- Non-null: 49 / 50
- Mean: -0.0105, Std: 1.0117
- Range: [-1.4301, 2.7202]
- Unique values: 49
### Feature_2 (float64)
- Non-null: 48 / 50
- Mean: 0.2735, Std: 1.0029
- Range: [-1.7630, 3.8527]
- Unique values: 48
### Feature_3 (float64)
- Non-null: 47 / 50
- Mean: 0.0442, Std: 0.6913
- Range: [-1.3202, 1.7655]
- Unique values: 47
### Feature_4 (float64)
- Non-null: 50 / 50
- Mean: -0.0598, Std: 1.0333
- Range: [-2.0251, 2.3147]
- Unique values: 50
### Feature_5 (float64)
- Non-null: 50 / 50
- Mean: 0.0901, Std: 0.9691
- Range: [-1.9876, 1.8968]
- Unique values: 50
### Feature_6 (float64)
- Non-null: 50 / 50
- Mean: -0.1957, Std: 0.8696
- Range: [-1.9597, 2.1905]
- Unique values: 50
### Feature_7 (float64)
- Non-null: 50 / 50
- Mean: -0.1685, Std: 1.0427
- Range: [-3.2413, 2.1532]
- Unique values: 50
### Feature_8 (float64)
- Non-null: 50 / 50
- Mean: 0.2135, Std: 1.0201
- Range: [-2.6197, 2.1898]
- Unique values: 50
### Target (float64)
- Non-null: 50 / 50
- Mean: 6.6353, Std: 135.6161
- Range: [-202.2208, 342.4230]
- Unique values: 50

## Correlation Matrix (numeric columns)
```
           Feature_1  Feature_2  Feature_3  Feature_4  Feature_5  Feature_6  Feature_7  Feature_8    Target
Feature_1   1.000000   0.311703   0.178468  -0.075462   0.039011  -0.190370  -0.008285  -0.152999  0.698307
Feature_2   0.311703   1.000000  -0.002471   0.168154   0.250303  -0.152128   0.048805  -0.119994  0.488613
Feature_3   0.178468  -0.002471   1.000000   0.022906  -0.052313   0.012453  -0.061907  -0.014087  0.328751
Feature_4  -0.075462   0.168154   0.022906   1.000000  -0.146289  -0.049277   0.026477  -0.135639  0.614038
Feature_5   0.039011   0.250303  -0.052313  -0.146289   1.000000   0.017106   0.094811   0.085956 -0.034305
Feature_6  -0.190370  -0.152128   0.012453  -0.049277   0.017106   1.000000  -0.037744   0.050316 -0.164833
Feature_7  -0.008285   0.048805  -0.061907   0.026477   0.094811  -0.037744   1.000000  -0.107362 -0.000195
Feature_8  -0.152999  -0.119994  -0.014087  -0.135639   0.085956   0.050316  -0.107362   1.000000 -0.164014
Target      0.698307   0.488613   0.328751   0.614038  -0.034305  -0.164833  -0.000195  -0.164014  1.000000
```