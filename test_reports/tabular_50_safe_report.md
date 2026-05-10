# EDA Report: tabular_50_safe.csv

## Overview
- Shape: 50 rows × 9 columns
- Data type: tabular
- Target column: Target

## Column Statistics
### Feature_1 (float64)
- Non-null: 50 / 50
- Mean: -0.0319, Std: 1.0126
- Range: [-1.4301, 2.7202]
- Unique values: 49
### Feature_2 (float64)
- Non-null: 50 / 50
- Mean: 0.2225, Std: 1.0153
- Range: [-1.7630, 3.8527]
- Unique values: 48
### Feature_3 (float64)
- Non-null: 50 / 50
- Mean: -0.0058, Std: 0.7007
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
Feature_1   1.000000   0.273169   0.203599  -0.056970   0.065288  -0.196569  -0.012864  -0.150438  0.701311
Feature_2   0.273169   1.000000  -0.052950   0.192890   0.196323  -0.132811   0.000312  -0.122097  0.449234
Feature_3   0.203599  -0.052950   1.000000  -0.056249  -0.092331   0.060439  -0.028300  -0.106230  0.286697
Feature_4  -0.056970   0.192890  -0.056249   1.000000  -0.146289  -0.049277   0.026477  -0.135639  0.614038
Feature_5   0.065288   0.196323  -0.092331  -0.146289   1.000000   0.017106   0.094811   0.085956 -0.034305
Feature_6  -0.196569  -0.132811   0.060439  -0.049277   0.017106   1.000000  -0.037744   0.050316 -0.164833
Feature_7  -0.012864   0.000312  -0.028300   0.026477   0.094811  -0.037744   1.000000  -0.107362 -0.000195
Feature_8  -0.150438  -0.122097  -0.106230  -0.135639   0.085956   0.050316  -0.107362   1.000000 -0.164014
Target      0.701311   0.449234   0.286697   0.614038  -0.034305  -0.164833  -0.000195  -0.164014  1.000000
```