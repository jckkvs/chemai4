# EDA Report: tabular_1000_large.csv

## Overview
- Shape: 1000 rows × 11 columns
- Data type: tabular
- Target column: Target

## Column Statistics
### Feature_1 (float64)
- Non-null: 1000 / 1000
- Mean: 50.1933, Std: 9.7922
- Range: [17.5873, 88.5273]
- Unique values: 1000
### Feature_2 (float64)
- Non-null: 1000 / 1000
- Mean: 50.7084, Std: 9.9745
- Range: [20.5961, 81.9311]
- Unique values: 1000
### Feature_3 (float64)
- Non-null: 1000 / 1000
- Mean: 50.0583, Std: 9.8345
- Range: [19.8049, 89.2624]
- Unique values: 1000
### Feature_4 (float64)
- Non-null: 1000 / 1000
- Mean: 49.8128, Std: 10.2713
- Range: [20.7055, 82.4309]
- Unique values: 1000
### Feature_5 (float64)
- Non-null: 1000 / 1000
- Mean: 49.5073, Std: 9.9238
- Range: [18.2330, 81.1291]
- Unique values: 1000
### Feature_6 (float64)
- Non-null: 1000 / 1000
- Mean: 49.5326, Std: 10.0739
- Range: [21.0049, 80.9830]
- Unique values: 1000
### Feature_7 (float64)
- Non-null: 1000 / 1000
- Mean: 49.7184, Std: 10.2500
- Range: [22.8739, 85.2906]
- Unique values: 1000
### Feature_8 (float64)
- Non-null: 1000 / 1000
- Mean: 50.1922, Std: 10.4174
- Range: [13.1163, 81.1768]
- Unique values: 1000
### Target (float64)
- Non-null: 1000 / 1000
- Mean: 100.3668, Std: 20.2337
- Range: [23.2669, 167.5477]
- Unique values: 1000
### Sample_ID (object)
- Non-null: 1000 / 1000
- Unique values: 1000
### Category (object)
- Non-null: 1000 / 1000
- Unique values: 5

## Correlation Matrix (numeric columns)
```
           Feature_1  Feature_2  Feature_3  Feature_4  Feature_5  Feature_6  Feature_7  Feature_8    Target
Feature_1   1.000000  -0.040400   0.022129  -0.013321  -0.031237  -0.005723  -0.032866   0.044947 -0.016585
Feature_2  -0.040400   1.000000  -0.011199  -0.054698  -0.018687   0.022221  -0.001289   0.056526  0.012436
Feature_3   0.022129  -0.011199   1.000000   0.021586   0.036015   0.003155  -0.029101  -0.039182  0.001739
Feature_4  -0.013321  -0.054698   0.021586   1.000000   0.019204   0.035692   0.011433  -0.003930  0.004894
Feature_5  -0.031237  -0.018687   0.036015   0.019204   1.000000  -0.042467   0.004021   0.028139 -0.023439
Feature_6  -0.005723   0.022221   0.003155   0.035692  -0.042467   1.000000   0.014822   0.056429  0.017869
Feature_7  -0.032866  -0.001289  -0.029101   0.011433   0.004021   0.014822   1.000000  -0.002537 -0.001001
Feature_8   0.044947   0.056526  -0.039182  -0.003930   0.028139   0.056429  -0.002537   1.000000  0.026523
Target     -0.016585   0.012436   0.001739   0.004894  -0.023439   0.017869  -0.001001   0.026523  1.000000
```