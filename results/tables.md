## Shift ladder on matched samples, 2 s horizon, PR-AUC [95% CI]

| rung                         | test          | subset    |   positives |   prevalence | rule              | boosting          | CNN               |
|:-----------------------------|:--------------|:----------|------------:|-------------:|:------------------|:------------------|:------------------|
| R0 highD->highD              | highd_test    | all       |         943 |        0.02  | 0.95 [0.94, 0.96] | 0.98 [0.98, 0.99] | 0.99 [0.98, 0.99] |
| R0 highD->highD              | highd_test    | pre_drift |          75 |        0.002 | 0.34 [0.22, 0.54] | 0.68 [0.54, 0.83] | 0.66 [0.55, 0.78] |
| R1 highD->highD new location | highd_new_loc | all       |        1008 |        0.017 | 0.96 [0.95, 0.97] | 0.98 [0.97, 0.99] | 0.99 [0.98, 0.99] |
| R1 highD->highD new location | highd_new_loc | pre_drift |          75 |        0.001 | 0.27 [0.18, 0.44] | 0.65 [0.54, 0.76] | 0.64 [0.51, 0.76] |
| R2 highD->exiD               | exid_new_loc  | all       |        6765 |        0.05  | 0.80 [0.77, 0.82] | 0.64 [0.62, 0.67] | 0.57 [0.55, 0.59] |
| R2 highD->exiD               | exid_new_loc  | pre_drift |         590 |        0.005 | 0.07 [0.06, 0.09] | 0.05 [0.04, 0.07] | 0.03 [0.02, 0.05] |
| R2 highD->exiD               | exid_seen_loc | all       |        3866 |        0.064 | 0.67 [0.63, 0.71] | 0.68 [0.64, 0.71] | 0.63 [0.58, 0.68] |
| R2 highD->exiD               | exid_seen_loc | pre_drift |         503 |        0.01  | 0.06 [0.04, 0.09] | 0.12 [0.07, 0.19] | 0.10 [0.07, 0.14] |
| R3 highD+1% exiD             | exid_new_loc  | all       |        6765 |        0.05  | 0.80 [0.77, 0.82] | 0.76 [0.74, 0.78] | 0.74 [0.72, 0.75] |
| R3 highD+1% exiD             | exid_new_loc  | pre_drift |         590 |        0.005 | 0.07 [0.06, 0.09] | 0.05 [0.03, 0.06] | 0.04 [0.03, 0.06] |
| R3 highD+1% exiD             | exid_seen_loc | all       |        3866 |        0.064 | 0.67 [0.63, 0.71] | 0.80 [0.77, 0.83] | 0.77 [0.73, 0.82] |
| R3 highD+1% exiD             | exid_seen_loc | pre_drift |         503 |        0.01  | 0.06 [0.04, 0.09] | 0.20 [0.14, 0.26] | 0.20 [0.15, 0.26] |
| R3 highD+5% exiD             | exid_new_loc  | all       |        6765 |        0.05  | 0.80 [0.77, 0.82] | 0.81 [0.80, 0.82] | 0.79 [0.77, 0.80] |
| R3 highD+5% exiD             | exid_new_loc  | pre_drift |         590 |        0.005 | 0.07 [0.06, 0.09] | 0.07 [0.05, 0.09] | 0.04 [0.03, 0.06] |
| R3 highD+5% exiD             | exid_seen_loc | all       |        3866 |        0.064 | 0.67 [0.63, 0.71] | 0.86 [0.83, 0.88] | 0.81 [0.79, 0.84] |
| R3 highD+5% exiD             | exid_seen_loc | pre_drift |         503 |        0.01  | 0.06 [0.04, 0.09] | 0.29 [0.22, 0.35] | 0.21 [0.15, 0.27] |
| R3 highD+20% exiD            | exid_new_loc  | all       |        6765 |        0.05  | 0.80 [0.77, 0.82] | 0.82 [0.81, 0.83] | 0.79 [0.78, 0.80] |
| R3 highD+20% exiD            | exid_new_loc  | pre_drift |         590 |        0.005 | 0.07 [0.06, 0.09] | 0.06 [0.05, 0.08] | 0.05 [0.04, 0.07] |
| R3 highD+20% exiD            | exid_seen_loc | all       |        3866 |        0.064 | 0.67 [0.63, 0.71] | 0.88 [0.86, 0.90] | 0.86 [0.84, 0.87] |
| R3 highD+20% exiD            | exid_seen_loc | pre_drift |         503 |        0.01  | 0.06 [0.04, 0.09] | 0.43 [0.34, 0.51] | 0.36 [0.26, 0.45] |
| R4 exiD->exiD                | exid_new_loc  | all       |        6765 |        0.05  | 0.80 [0.77, 0.82] | 0.83 [0.82, 0.84] | 0.80 [0.78, 0.81] |
| R4 exiD->exiD                | exid_new_loc  | pre_drift |         590 |        0.005 | 0.07 [0.06, 0.09] | 0.05 [0.04, 0.07] | 0.07 [0.05, 0.09] |
| R4 exiD->exiD                | exid_seen_loc | all       |        3866 |        0.064 | 0.67 [0.63, 0.71] | 0.91 [0.89, 0.92] | 0.90 [0.88, 0.91] |
| R4 exiD->exiD                | exid_seen_loc | pre_drift |         503 |        0.01  | 0.06 [0.04, 0.09] | 0.56 [0.46, 0.64] | 0.51 [0.43, 0.57] |

## Paired differences at 2 s (Holm-adjusted over all cells)

| rung                         | test          | subset    |   positives | CNN - boosting          |   p (Holm) | boosting - rule         |   p (Holm)  |
|:-----------------------------|:--------------|:----------|------------:|:------------------------|-----------:|:------------------------|------------:|
| R0 highD->highD              | highd_test    | all       |         943 | +0.001 [-0.001, +0.003] |      1     | +0.031 [+0.025, +0.037] |       0.01  |
| R0 highD->highD              | highd_test    | pre_drift |          75 | -0.020 [-0.068, +0.024] |      1     | +0.338 [+0.219, +0.449] |       0.01  |
| R1 highD->highD new location | highd_new_loc | all       |        1008 | +0.003 [+0.000, +0.006] |      0.292 | +0.026 [+0.020, +0.031] |       0.01  |
| R1 highD->highD new location | highd_new_loc | pre_drift |          75 | -0.012 [-0.058, +0.038] |      1     | +0.385 [+0.269, +0.461] |       0.01  |
| R2 highD->exiD               | exid_new_loc  | all       |        6765 | -0.073 [-0.087, -0.059] |      0.01  | -0.155 [-0.169, -0.142] |       0.01  |
| R2 highD->exiD               | exid_new_loc  | pre_drift |         590 | -0.017 [-0.029, -0.008] |      0.01  | -0.024 [-0.038, -0.010] |       0.01  |
| R2 highD->exiD               | exid_seen_loc | all       |        3866 | -0.049 [-0.075, -0.020] |      0.024 | +0.004 [-0.027, +0.037] |       1     |
| R2 highD->exiD               | exid_seen_loc | pre_drift |         503 | -0.016 [-0.053, +0.009] |      1     | +0.061 [+0.014, +0.126] |       0.058 |
| R3 highD+1% exiD             | exid_new_loc  | all       |        6765 | -0.024 [-0.040, -0.009] |      0.01  | -0.034 [-0.049, -0.019] |       0.01  |
| R3 highD+1% exiD             | exid_new_loc  | pre_drift |         590 | -0.002 [-0.016, +0.015] |      1     | -0.029 [-0.046, -0.012] |       0.018 |
| R3 highD+1% exiD             | exid_seen_loc | all       |        3866 | -0.028 [-0.042, -0.008] |      0.053 | +0.127 [+0.107, +0.147] |       0.01  |
| R3 highD+1% exiD             | exid_seen_loc | pre_drift |         503 | +0.001 [-0.032, +0.030] |      1     | +0.139 [+0.088, +0.198] |       0.01  |
| R3 highD+5% exiD             | exid_new_loc  | all       |        6765 | -0.023 [-0.033, -0.013] |      0.01  | +0.013 [-0.007, +0.035] |       0.907 |
| R3 highD+5% exiD             | exid_new_loc  | pre_drift |         590 | -0.025 [-0.040, -0.013] |      0.01  | -0.005 [-0.023, +0.015] |       1     |
| R3 highD+5% exiD             | exid_seen_loc | all       |        3866 | -0.043 [-0.056, -0.030] |      0.01  | +0.187 [+0.155, +0.219] |       0.01  |
| R3 highD+5% exiD             | exid_seen_loc | pre_drift |         503 | -0.080 [-0.125, -0.034] |      0.01  | +0.229 [+0.172, +0.290] |       0.01  |
| R3 highD+20% exiD            | exid_new_loc  | all       |        6765 | -0.029 [-0.037, -0.022] |      0.01  | +0.024 [-0.002, +0.051] |       0.362 |
| R3 highD+20% exiD            | exid_new_loc  | pre_drift |         590 | -0.011 [-0.026, +0.004] |      0.969 | -0.010 [-0.028, +0.010] |       1     |
| R3 highD+20% exiD            | exid_seen_loc | all       |        3866 | -0.026 [-0.032, -0.017] |      0.01  | +0.212 [+0.174, +0.251] |       0.01  |
| R3 highD+20% exiD            | exid_seen_loc | pre_drift |         503 | -0.074 [-0.123, -0.013] |      0.216 | +0.375 [+0.294, +0.439] |       0.01  |
| R4 exiD->exiD                | exid_new_loc  | all       |        6765 | -0.032 [-0.040, -0.023] |      0.01  | +0.034 [+0.006, +0.064] |       0.07  |
| R4 exiD->exiD                | exid_new_loc  | pre_drift |         590 | +0.015 [+0.001, +0.033] |      0.292 | -0.020 [-0.037, -0.005] |       0.059 |
| R4 exiD->exiD                | exid_seen_loc | all       |        3866 | -0.013 [-0.019, -0.005] |      0.021 | +0.238 [+0.196, +0.281] |       0.01  |
| R4 exiD->exiD                | exid_seen_loc | pre_drift |         503 | -0.053 [-0.076, -0.028] |      0.011 | +0.505 [+0.421, +0.564] |       0.01  |

## Shuffled-label controls (2 s)

| trained on   | test          | subset    |   prevalence | boosting, shuffled   |   CNN shuffled: mean | CNN shuffled: per seed              |
|:-------------|:--------------|:----------|-------------:|:---------------------|---------------------:|:------------------------------------|
| highD        | highd_test    | all       |        0.02  | 0.01 [0.01, 0.02]    |                0.065 | [0.014, 0.014, 0.011, 0.022, 0.263] |
| highD        | highd_test    | pre_drift |        0.002 | 0.00 [0.00, 0.00]    |                0.004 | [0.009, 0.002, 0.001, 0.005, 0.002] |
| highD        | highd_new_loc | all       |        0.017 | 0.01 [0.01, 0.02]    |                0.084 | [0.011, 0.01, 0.009, 0.015, 0.376]  |
| highD        | highd_new_loc | pre_drift |        0.001 | 0.01 [0.00, 0.04]    |                0.001 | [0.002, 0.001, 0.001, 0.002, 0.002] |
| highD        | exid_seen_loc | all       |        0.064 | 0.05 [0.04, 0.05]    |                0.141 | [0.05, 0.055, 0.041, 0.095, 0.464]  |
| highD        | exid_seen_loc | pre_drift |        0.01  | 0.01 [0.01, 0.02]    |                0.013 | [0.009, 0.007, 0.009, 0.02, 0.02]   |
| highD        | exid_new_loc  | all       |        0.05  | 0.03 [0.03, 0.03]    |                0.13  | [0.034, 0.045, 0.03, 0.074, 0.468]  |
| highD        | exid_new_loc  | pre_drift |        0.005 | 0.00 [0.00, 0.00]    |                0.007 | [0.004, 0.005, 0.005, 0.013, 0.01]  |
| exiD         | exid_seen_loc | all       |        0.064 | 0.05 [0.04, 0.05]    |                0.108 | [0.096, 0.06, 0.035, 0.092, 0.255]  |
| exiD         | exid_seen_loc | pre_drift |        0.01  | 0.01 [0.01, 0.02]    |                0.01  | [0.009, 0.008, 0.006, 0.013, 0.013] |
| exiD         | exid_new_loc  | all       |        0.05  | 0.03 [0.03, 0.03]    |                0.107 | [0.094, 0.051, 0.027, 0.085, 0.28]  |
| exiD         | exid_new_loc  | pre_drift |        0.005 | 0.00 [0.00, 0.00]    |                0.005 | [0.005, 0.004, 0.004, 0.007, 0.006] |

## Leave-one-location-out (boosting, 2 s)

| dataset   |   held-out location | subset    |   recordings |   positives | rule              | boosting          |
|:----------|--------------------:|:----------|-------------:|------------:|:------------------|:------------------|
| highd     |                   1 | all       |           37 |        3732 | 0.96 [0.95, 0.96] | 0.98 [0.98, 0.98] |
| highd     |                   1 | pre_drift |           37 |         282 | 0.43 [0.36, 0.51] | 0.63 [0.56, 0.69] |
| highd     |                   2 | all       |            3 |         315 | 0.97 [0.96, 0.98] | 0.99 [0.99, 1.00] |
| highd     |                   2 | pre_drift |            3 |          21 | 0.38 [0.37, 0.48] | 0.71 [0.60, 0.95] |
| highd     |                   3 | all       |            3 |         567 | 0.94 [0.93, 0.96] | 0.97 [0.96, 0.98] |
| highd     |                   3 | pre_drift |            3 |          54 | 0.45 [0.42, 0.51] | 0.67 [0.60, 0.74] |
| highd     |                   4 | all       |            4 |         660 | 0.94 [0.92, 0.96] | 0.97 [0.95, 0.98] |
| highd     |                   4 | pre_drift |            4 |          52 | 0.27 [0.17, 0.48] | 0.52 [0.37, 0.72] |
| highd     |                   5 | all       |           10 |        1008 | 0.96 [0.95, 0.97] | 0.98 [0.97, 0.99] |
| highd     |                   5 | pre_drift |           10 |          75 | 0.27 [0.18, 0.45] | 0.62 [0.46, 0.76] |
| highd     |                   6 | all       |            3 |         518 | 0.90 [0.89, 0.91] | 0.96 [0.96, 0.97] |
| highd     |                   6 | pre_drift |            3 |          81 | 0.49 [0.38, 0.58] | 0.80 [0.67, 0.90] |
| exid      |                   0 | all       |           19 |        4736 | 0.68 [0.64, 0.73] | 0.82 [0.80, 0.84] |
| exid      |                   0 | pre_drift |           19 |         374 | 0.04 [0.03, 0.05] | 0.15 [0.11, 0.19] |
| exid      |                   1 | all       |           20 |        7960 | 0.58 [0.56, 0.60] | 0.79 [0.77, 0.80] |
| exid      |                   1 | pre_drift |           20 |        1110 | 0.10 [0.09, 0.12] | 0.23 [0.20, 0.27] |
| exid      |                   2 | all       |           14 |        5438 | 0.70 [0.69, 0.71] | 0.87 [0.86, 0.88] |
| exid      |                   2 | pre_drift |           14 |         804 | 0.06 [0.05, 0.08] | 0.29 [0.26, 0.33] |
| exid      |                   3 | all       |            8 |        2591 | 0.72 [0.70, 0.75] | 0.85 [0.84, 0.86] |
| exid      |                   3 | pre_drift |            8 |         201 | 0.06 [0.04, 0.09] | 0.05 [0.03, 0.10] |
| exid      |                   4 | all       |           12 |        2832 | 0.77 [0.75, 0.80] | 0.90 [0.88, 0.91] |
| exid      |                   4 | pre_drift |           12 |         195 | 0.08 [0.06, 0.09] | 0.15 [0.13, 0.20] |
| exid      |                   5 | all       |            5 |        2239 | 0.80 [0.79, 0.82] | 0.91 [0.90, 0.92] |
| exid      |                   5 | pre_drift |            5 |          57 | 0.02 [0.01, 0.04] | 0.12 [0.06, 0.20] |
| exid      |                   6 | all       |           15 |        4174 | 0.84 [0.83, 0.86] | 0.83 [0.82, 0.84] |
| exid      |                   6 | pre_drift |           15 |         389 | 0.09 [0.07, 0.12] | 0.07 [0.06, 0.09] |

## Sensitivity to label and pre-drift definitions (boosting, 2 s)

| variant       | rung                    | subset    |   positives | rule              | boosting          |
|:--------------|:------------------------|:----------|------------:|:------------------|:------------------|
| base          | R0 highD->highD (test)  | all       |         943 | 0.95 [0.94, 0.96] | 0.98 [0.98, 0.99] |
| base          | R0 highD->highD (test)  | pre_drift |          75 | 0.34 [0.22, 0.53] | 0.68 [0.54, 0.83] |
| base          | R2 highD->exiD seen loc | all       |        3866 | 0.67 [0.63, 0.71] | 0.68 [0.64, 0.71] |
| base          | R2 highD->exiD seen loc | pre_drift |         503 | 0.06 [0.04, 0.09] | 0.12 [0.07, 0.19] |
| base          | R2 highD->exiD new loc  | all       |        6765 | 0.80 [0.77, 0.82] | 0.64 [0.62, 0.67] |
| base          | R2 highD->exiD new loc  | pre_drift |         590 | 0.07 [0.06, 0.09] | 0.05 [0.03, 0.07] |
| base          | R4 exiD->exiD seen loc  | all       |        3866 | 0.67 [0.63, 0.71] | 0.91 [0.89, 0.92] |
| base          | R4 exiD->exiD seen loc  | pre_drift |         503 | 0.06 [0.04, 0.09] | 0.56 [0.47, 0.64] |
| base          | R4 exiD->exiD new loc   | all       |        6765 | 0.80 [0.77, 0.82] | 0.83 [0.82, 0.84] |
| base          | R4 exiD->exiD new loc   | pre_drift |         590 | 0.07 [0.06, 0.09] | 0.05 [0.04, 0.07] |
| cutoff 0.15 m | R0 highD->highD (test)  | pre_drift |          13 | 0.18 [0.02, 0.42] | 0.55 [0.28, 0.79] |
| cutoff 0.15 m | R2 highD->exiD seen loc | pre_drift |         112 | 0.01 [0.01, 0.02] | 0.03 [0.02, 0.08] |
| cutoff 0.15 m | R2 highD->exiD new loc  | pre_drift |         116 | 0.03 [0.01, 0.07] | 0.01 [0.00, 0.01] |
| cutoff 0.15 m | R4 exiD->exiD seen loc  | pre_drift |         112 | 0.01 [0.01, 0.02] | 0.44 [0.33, 0.56] |
| cutoff 0.15 m | R4 exiD->exiD new loc   | pre_drift |         116 | 0.03 [0.01, 0.07] | 0.01 [0.00, 0.01] |
| cutoff 0.6 m  | R0 highD->highD (test)  | pre_drift |         487 | 0.86 [0.81, 0.91] | 0.95 [0.92, 0.97] |
| cutoff 0.6 m  | R2 highD->exiD seen loc | pre_drift |        2137 | 0.37 [0.34, 0.41] | 0.41 [0.35, 0.48] |
| cutoff 0.6 m  | R2 highD->exiD new loc  | pre_drift |        3476 | 0.56 [0.53, 0.59] | 0.39 [0.35, 0.43] |
| cutoff 0.6 m  | R4 exiD->exiD seen loc  | pre_drift |        2137 | 0.37 [0.34, 0.41] | 0.81 [0.74, 0.85] |
| cutoff 0.6 m  | R4 exiD->exiD new loc   | pre_drift |        3476 | 0.56 [0.53, 0.59] | 0.60 [0.58, 0.63] |
| disp 1.2 m    | R0 highD->highD (test)  | all       |         962 | 0.95 [0.94, 0.96] | 0.98 [0.97, 0.99] |
| disp 1.2 m    | R0 highD->highD (test)  | pre_drift |          85 | 0.33 [0.21, 0.51] | 0.69 [0.60, 0.79] |
| disp 1.2 m    | R2 highD->exiD seen loc | all       |        3765 | 0.66 [0.62, 0.71] | 0.66 [0.64, 0.69] |
| disp 1.2 m    | R2 highD->exiD seen loc | pre_drift |         483 | 0.07 [0.05, 0.10] | 0.12 [0.08, 0.17] |
| disp 1.2 m    | R2 highD->exiD new loc  | all       |        6859 | 0.78 [0.74, 0.81] | 0.60 [0.58, 0.62] |
| disp 1.2 m    | R2 highD->exiD new loc  | pre_drift |         699 | 0.08 [0.06, 0.10] | 0.03 [0.02, 0.05] |
| disp 1.2 m    | R4 exiD->exiD seen loc  | all       |        3765 | 0.66 [0.62, 0.71] | 0.90 [0.87, 0.91] |
| disp 1.2 m    | R4 exiD->exiD seen loc  | pre_drift |         483 | 0.07 [0.05, 0.10] | 0.53 [0.43, 0.61] |
| disp 1.2 m    | R4 exiD->exiD new loc   | all       |        6859 | 0.78 [0.74, 0.81] | 0.81 [0.80, 0.82] |
| disp 1.2 m    | R4 exiD->exiD new loc   | pre_drift |         699 | 0.08 [0.06, 0.10] | 0.05 [0.04, 0.07] |
| disp 2.4 m    | R0 highD->highD (test)  | all       |         945 | 0.96 [0.95, 0.97] | 0.98 [0.98, 0.99] |
| disp 2.4 m    | R0 highD->highD (test)  | pre_drift |          71 | 0.34 [0.27, 0.44] | 0.62 [0.55, 0.69] |
| disp 2.4 m    | R2 highD->exiD seen loc | all       |        3526 | 0.69 [0.64, 0.74] | 0.70 [0.68, 0.72] |
| disp 2.4 m    | R2 highD->exiD seen loc | pre_drift |         365 | 0.05 [0.03, 0.09] | 0.15 [0.10, 0.22] |
| disp 2.4 m    | R2 highD->exiD new loc  | all       |        6156 | 0.80 [0.77, 0.84] | 0.63 [0.61, 0.65] |
| disp 2.4 m    | R2 highD->exiD new loc  | pre_drift |         402 | 0.07 [0.05, 0.09] | 0.04 [0.02, 0.05] |
| disp 2.4 m    | R4 exiD->exiD seen loc  | all       |        3526 | 0.69 [0.64, 0.74] | 0.92 [0.90, 0.94] |
| disp 2.4 m    | R4 exiD->exiD seen loc  | pre_drift |         365 | 0.05 [0.03, 0.09] | 0.57 [0.47, 0.65] |
| disp 2.4 m    | R4 exiD->exiD new loc   | all       |        6156 | 0.80 [0.77, 0.84] | 0.84 [0.83, 0.85] |
| disp 2.4 m    | R4 exiD->exiD new loc   | pre_drift |         402 | 0.07 [0.05, 0.09] | 0.07 [0.05, 0.09] |
| window 3 s    | R0 highD->highD (test)  | all       |         959 | 0.95 [0.93, 0.96] | 0.98 [0.98, 0.99] |
| window 3 s    | R0 highD->highD (test)  | pre_drift |          94 | 0.44 [0.34, 0.60] | 0.76 [0.69, 0.84] |
| window 3 s    | R2 highD->exiD seen loc | all       |        3813 | 0.67 [0.63, 0.71] | 0.64 [0.61, 0.67] |
| window 3 s    | R2 highD->exiD seen loc | pre_drift |         473 | 0.07 [0.05, 0.10] | 0.11 [0.07, 0.16] |
| window 3 s    | R2 highD->exiD new loc  | all       |        6774 | 0.81 [0.78, 0.84] | 0.63 [0.61, 0.66] |
| window 3 s    | R2 highD->exiD new loc  | pre_drift |         600 | 0.09 [0.07, 0.11] | 0.05 [0.04, 0.07] |
| window 3 s    | R4 exiD->exiD seen loc  | all       |        3813 | 0.67 [0.63, 0.71] | 0.91 [0.89, 0.93] |
| window 3 s    | R4 exiD->exiD seen loc  | pre_drift |         473 | 0.07 [0.05, 0.10] | 0.56 [0.48, 0.63] |
| window 3 s    | R4 exiD->exiD new loc   | all       |        6774 | 0.81 [0.78, 0.84] | 0.83 [0.82, 0.85] |
| window 3 s    | R4 exiD->exiD new loc   | pre_drift |         600 | 0.09 [0.07, 0.11] | 0.06 [0.05, 0.08] |
| window 5 s    | R0 highD->highD (test)  | all       |         918 | 0.94 [0.92, 0.96] | 0.97 [0.96, 0.98] |
| window 5 s    | R0 highD->highD (test)  | pre_drift |          86 | 0.40 [0.32, 0.52] | 0.64 [0.53, 0.77] |
| window 5 s    | R2 highD->exiD seen loc | all       |        3688 | 0.65 [0.61, 0.70] | 0.68 [0.66, 0.70] |
| window 5 s    | R2 highD->exiD seen loc | pre_drift |         458 | 0.06 [0.05, 0.09] | 0.12 [0.08, 0.16] |
| window 5 s    | R2 highD->exiD new loc  | all       |        6261 | 0.78 [0.74, 0.81] | 0.64 [0.61, 0.67] |
| window 5 s    | R2 highD->exiD new loc  | pre_drift |         516 | 0.08 [0.06, 0.10] | 0.03 [0.02, 0.05] |
| window 5 s    | R4 exiD->exiD seen loc  | all       |        3688 | 0.65 [0.61, 0.70] | 0.89 [0.87, 0.91] |
| window 5 s    | R4 exiD->exiD seen loc  | pre_drift |         458 | 0.06 [0.05, 0.09] | 0.54 [0.44, 0.61] |
| window 5 s    | R4 exiD->exiD new loc   | all       |        6261 | 0.78 [0.74, 0.81] | 0.80 [0.79, 0.81] |
| window 5 s    | R4 exiD->exiD new loc   | pre_drift |         516 | 0.08 [0.06, 0.10] | 0.05 [0.04, 0.06] |

## Ensembles and hierarchy at 2 s (difference to boosting, Holm p over 60 tests)

| config                            | test          | subset    |   positives |   rule | boosting          |   CNN (3-seed mean score) | hierarchy              | stack: rule+boost+CNN   | stack + specialist     |
|:----------------------------------|:--------------|:----------|------------:|-------:|:------------------|--------------------------:|:-----------------------|:------------------------|:-----------------------|
| R0/R1 highD (source-calibrated)   | highd_test    | all       |         943 |   0.95 | 0.98 [0.98, 0.99] |                      0.99 | 0.98 (-0.003, p=0.248) | 0.99 (+0.003, p=0.322)  | 0.99 (+0.003, p=0.330) |
| R0/R1 highD (source-calibrated)   | highd_test    | pre_drift |          75 |   0.34 | 0.68 [0.54, 0.83] |                      0.67 | 0.59 (-0.092, p=0.534) | 0.68 (-0.003, p=1.000)  | 0.68 (-0.000, p=1.000) |
| R0/R1 highD (source-calibrated)   | highd_new_loc | all       |        1008 |   0.96 | 0.98 [0.97, 0.99] |                      0.99 | 0.98 (-0.003, p=0.024) | 0.99 (+0.005, p=0.024)  | 0.99 (+0.005, p=0.024) |
| R0/R1 highD (source-calibrated)   | highd_new_loc | pre_drift |          75 |   0.27 | 0.65 [0.54, 0.76] |                      0.67 | 0.55 (-0.098, p=0.024) | 0.68 (+0.027, p=1.000)  | 0.69 (+0.041, p=1.000) |
| R2 highD->exiD, source-calibrated | exid_seen_loc | all       |        3866 |   0.67 | 0.68 [0.64, 0.71] |                      0.64 | 0.66 (-0.019, p=0.024) | 0.68 (+0.009, p=1.000)  | 0.70 (+0.021, p=1.000) |
| R2 highD->exiD, source-calibrated | exid_seen_loc | pre_drift |         503 |   0.06 | 0.12 [0.07, 0.19] |                      0.11 | 0.12 (-0.001, p=1.000) | 0.11 (-0.010, p=1.000)  | 0.11 (-0.010, p=1.000) |
| R2 highD->exiD, source-calibrated | exid_new_loc  | all       |        6765 |   0.8  | 0.64 [0.62, 0.67] |                      0.59 | 0.64 (+0.003, p=1.000) | 0.64 (+0.002, p=1.000)  | 0.64 (-0.002, p=1.000) |
| R2 highD->exiD, source-calibrated | exid_new_loc  | pre_drift |         590 |   0.07 | 0.05 [0.04, 0.07] |                      0.04 | 0.03 (-0.019, p=0.024) | 0.04 (-0.010, p=1.000)  | 0.04 (-0.014, p=0.104) |
| R2 highD->exiD, target-calibrated | exid_seen_loc | all       |        3866 |   0.67 | 0.68 [0.64, 0.71] |                      0.64 | 0.70 (+0.025, p=0.024) | 0.72 (+0.041, p=0.104)  | 0.74 (+0.062, p=0.024) |
| R2 highD->exiD, target-calibrated | exid_seen_loc | pre_drift |         503 |   0.06 | 0.12 [0.07, 0.19] |                      0.11 | 0.11 (-0.007, p=1.000) | 0.12 (+0.002, p=1.000)  | 0.12 (-0.004, p=1.000) |
| R2 highD->exiD, target-calibrated | exid_new_loc  | all       |        6765 |   0.8  | 0.64 [0.62, 0.67] |                      0.59 | 0.72 (+0.075, p=0.024) | 0.78 (+0.135, p=0.024)  | 0.78 (+0.142, p=0.024) |
| R2 highD->exiD, target-calibrated | exid_new_loc  | pre_drift |         590 |   0.07 | 0.05 [0.04, 0.07] |                      0.04 | 0.03 (-0.016, p=0.026) | 0.06 (+0.007, p=1.000)  | 0.05 (+0.004, p=1.000) |
| R3 +5% exiD, target-calibrated    | exid_seen_loc | all       |        3866 |   0.67 | 0.86 [0.83, 0.88] |                      0.83 | 0.85 (-0.007, p=0.024) | 0.86 (+0.006, p=0.024)  | 0.86 (+0.006, p=0.024) |
| R3 +5% exiD, target-calibrated    | exid_seen_loc | pre_drift |         503 |   0.06 | 0.29 [0.22, 0.35] |                      0.23 | 0.23 (-0.058, p=0.024) | 0.30 (+0.007, p=1.000)  | 0.30 (+0.008, p=1.000) |
| R3 +5% exiD, target-calibrated    | exid_new_loc  | all       |        6765 |   0.8  | 0.81 [0.80, 0.82] |                      0.8  | 0.82 (+0.006, p=0.330) | 0.83 (+0.016, p=0.024)  | 0.82 (+0.014, p=0.024) |
| R3 +5% exiD, target-calibrated    | exid_new_loc  | pre_drift |         590 |   0.07 | 0.07 [0.05, 0.09] |                      0.05 | 0.02 (-0.047, p=0.024) | 0.07 (-0.004, p=1.000)  | 0.06 (-0.008, p=1.000) |
| R4 exiD only, target-calibrated   | exid_seen_loc | all       |        3866 |   0.67 | 0.91 [0.89, 0.92] |                      0.9  | 0.90 (-0.006, p=0.024) | 0.92 (+0.006, p=0.024)  | 0.91 (+0.006, p=0.050) |
| R4 exiD only, target-calibrated   | exid_seen_loc | pre_drift |         503 |   0.06 | 0.56 [0.46, 0.64] |                      0.55 | 0.50 (-0.059, p=0.024) | 0.61 (+0.042, p=0.024)  | 0.60 (+0.041, p=0.024) |
| R4 exiD only, target-calibrated   | exid_new_loc  | all       |        6765 |   0.8  | 0.83 [0.82, 0.84] |                      0.82 | 0.83 (-0.005, p=0.024) | 0.84 (+0.010, p=0.024)  | 0.84 (+0.011, p=0.024) |
| R4 exiD only, target-calibrated   | exid_new_loc  | pre_drift |         590 |   0.07 | 0.05 [0.04, 0.07] |                      0.09 | 0.04 (-0.014, p=0.084) | 0.09 (+0.037, p=0.024)  | 0.09 (+0.037, p=0.024) |

Specialists:

| training set   |   specialist leaves |   specialist lr |   specialist val AP (pre-drift) |   specialist train positives | train domains     |
|:---------------|--------------------:|----------------:|--------------------------------:|-----------------------------:|:------------------|
| exid           |                  16 |            0.1  |                          0.4316 |                         1473 | ['exid']          |
| k0.05          |                  16 |            0.06 |                          0.2632 |                          406 | ['highd', 'exid'] |
| src            |                  16 |            0.03 |                          0.635  |                          348 | ['highd']         |

## 13 ensemble and hierarchy variants, selected on the calibration split

Calibration cross-validation score per variant (higher is better; selection uses only this):

|             |   R0/R1 highD (source-cal) |   R2 highD->exiD (source-cal) |   R2 highD->exiD (target-cal) |   R3 +5% exiD (target-cal) |   R4 exiD only (target-cal) |
|:------------|---------------------------:|------------------------------:|------------------------------:|---------------------------:|----------------------------:|
| gen         |                      0.787 |                         0.787 |                         0.438 |                      0.602 |                       0.677 |
| avg         |                      0.743 |                         0.743 |                         0.439 |                      0.563 |                       0.627 |
| wavg        |                      0.756 |                         0.756 |                         0.442 |                      0.607 |                       0.701 |
| lr_flat     |                      0.83  |                         0.83  |                         0.43  |                      0.61  |                       0.707 |
| lr_regime   |                      0.828 |                         0.828 |                         0.436 |                      0.608 |                       0.706 |
| lr_split    |                      0.825 |                         0.825 |                         0.436 |                      0.608 |                       0.705 |
| gbm_stack   |                      0.818 |                         0.818 |                         0.428 |                      0.607 |                       0.703 |
| lr_wide     |                      0.822 |                         0.822 |                         0.498 |                      0.606 |                       0.706 |
| soft_gate   |                      0.778 |                         0.778 |                         0.462 |                      0.598 |                       0.677 |
| hier_0.15   |                      0.766 |                         0.766 |                         0.443 |                      0.581 |                       0.66  |
| hier_0.3    |                      0.717 |                         0.717 |                         0.437 |                      0.556 |                       0.657 |
| hier_0.6    |                      0.755 |                         0.755 |                         0.437 |                      0.582 |                       0.66  |
| cascade_res |                      0.743 |                         0.743 |                         0.429 |                      0.557 |                       0.655 |

Selected variant tested against plain boosting (paired bootstrap, Holm over 20 cells):

| config                      | test          | subset    |   positives | selected   |   boosting AP | selected - boosting     |   p (Holm) |
|:----------------------------|:--------------|:----------|------------:|:-----------|--------------:|:------------------------|-----------:|
| R0/R1 highD (source-cal)    | highd_test    | all       |         943 | lr_flat    |         0.984 | +0.003 [+0.001, +0.005] |      0.05  |
| R0/R1 highD (source-cal)    | highd_test    | pre_drift |          75 | lr_flat    |         0.682 | +0.009 [-0.024, +0.045] |      1     |
| R0/R1 highD (source-cal)    | highd_new_loc | all       |        1008 | lr_flat    |         0.983 | +0.005 [+0.002, +0.008] |      0.008 |
| R0/R1 highD (source-cal)    | highd_new_loc | pre_drift |          75 | lr_flat    |         0.652 | +0.038 [-0.004, +0.073] |      0.42  |
| R2 highD->exiD (source-cal) | exid_seen_loc | all       |        3866 | lr_flat    |         0.675 | -0.036 [-0.058, -0.011] |      0.048 |
| R2 highD->exiD (source-cal) | exid_seen_loc | pre_drift |         503 | lr_flat    |         0.12  | -0.005 [-0.034, +0.014] |      1     |
| R2 highD->exiD (source-cal) | exid_new_loc  | all       |        6765 | lr_flat    |         0.641 | -0.049 [-0.060, -0.036] |      0.008 |
| R2 highD->exiD (source-cal) | exid_new_loc  | pre_drift |         590 | lr_flat    |         0.05  | -0.012 [-0.023, -0.002] |      0.131 |
| R2 highD->exiD (target-cal) | exid_seen_loc | all       |        3866 | lr_wide    |         0.675 | +0.076 [+0.042, +0.107] |      0.008 |
| R2 highD->exiD (target-cal) | exid_seen_loc | pre_drift |         503 | lr_wide    |         0.12  | +0.032 [+0.009, +0.051] |      0.04  |
| R2 highD->exiD (target-cal) | exid_new_loc  | all       |        6765 | lr_wide    |         0.641 | +0.122 [+0.102, +0.142] |      0.008 |
| R2 highD->exiD (target-cal) | exid_new_loc  | pre_drift |         590 | lr_wide    |         0.05  | +0.009 [-0.002, +0.019] |      0.56  |
| R3 +5% exiD (target-cal)    | exid_seen_loc | all       |        3866 | lr_flat    |         0.858 | +0.005 [+0.001, +0.009] |      0.204 |
| R3 +5% exiD (target-cal)    | exid_seen_loc | pre_drift |         503 | lr_flat    |         0.288 | +0.015 [-0.007, +0.035] |      0.733 |
| R3 +5% exiD (target-cal)    | exid_new_loc  | all       |        6765 | lr_flat    |         0.809 | +0.017 [+0.011, +0.022] |      0.008 |
| R3 +5% exiD (target-cal)    | exid_new_loc  | pre_drift |         590 | lr_flat    |         0.069 | -0.002 [-0.012, +0.007] |      1     |
| R4 exiD only (target-cal)   | exid_seen_loc | all       |        3866 | lr_flat    |         0.909 | +0.006 [+0.002, +0.010] |      0.008 |
| R4 exiD only (target-cal)   | exid_seen_loc | pre_drift |         503 | lr_flat    |         0.564 | +0.039 [+0.030, +0.048] |      0.008 |
| R4 exiD only (target-cal)   | exid_new_loc  | all       |        6765 | lr_flat    |         0.83  | +0.015 [+0.009, +0.022] |      0.008 |
| R4 exiD only (target-cal)   | exid_new_loc  | pre_drift |         590 | lr_flat    |         0.054 | +0.038 [+0.022, +0.062] |      0.008 |

All variants on test, descriptive only (no significance claims):

| variant     |   all: mean diff to boosting | all: cells better   |   pre_drift: mean diff to boosting | pre_drift: cells better   |
|:------------|-----------------------------:|:--------------------|-----------------------------------:|:--------------------------|
| avg         |                        0.022 | 6/10                |                             -0.029 | 6/10                      |
| wavg        |                        0.014 | 9/10                |                              0.013 | 7/10                      |
| lr_flat     |                        0.012 | 8/10                |                              0.013 | 6/10                      |
| lr_regime   |                        0.023 | 10/10               |                              0.01  | 6/10                      |
| lr_split    |                        0.025 | 10/10               |                              0.01  | 6/10                      |
| gbm_stack   |                        0.015 | 8/10                |                              0.009 | 5/10                      |
| lr_wide     |                        0.024 | 9/10                |                              0.013 | 6/10                      |
| soft_gate   |                        0.013 | 7/10                |                             -0.003 | 5/10                      |
| hier_0.15   |                        0.004 | 4/10                |                             -0.008 | 5/10                      |
| hier_0.3    |                        0.006 | 4/10                |                             -0.039 | 0/10                      |
| hier_0.6    |                       -0.004 | 3/10                |                             -0.007 | 4/10                      |
| cascade_res |                        0.007 | 4/10                |                             -0.037 | 0/10                      |

## Conformal prediction (alpha 0.1, 2 s horizon)

| scenario                               | test          | subset    |   test positives | coverage (all)    | coverage (lane changes)   |   mean set size | empty sets   |
|:---------------------------------------|:--------------|:----------|-----------------:|:------------------|:--------------------------|----------------:|:-------------|
| exchangeable: highD cal -> highD       | highd_test    | all       |              943 | 0.90 [0.87, 0.93] | 0.88 [0.85, 0.92]         |            0.9  | 0%           |
| exchangeable: highD cal -> highD       | highd_test    | pre_drift |               75 | 0.91 [0.87, 0.94] | 0.29 [0.19, 0.45]         |            0.91 | 0%           |
| exchangeable: highD cal -> highD       | highd_new_loc | all       |             1008 | 0.89 [0.85, 0.91] | 0.89 [0.84, 0.92]         |            0.89 | 0%           |
| exchangeable: highD cal -> highD       | highd_new_loc | pre_drift |               75 | 0.89 [0.86, 0.92] | 0.36 [0.19, 0.55]         |            0.89 | 0%           |
| shift: highD cal -> exiD               | exid_seen_loc | all       |             3866 | 0.58 [0.48, 0.67] | 0.66 [0.60, 0.72]         |            0.6  | 0%           |
| shift: highD cal -> exiD               | exid_seen_loc | pre_drift |              503 | 0.60 [0.49, 0.70] | 0.13 [0.09, 0.19]         |            0.61 | 0%           |
| shift: highD cal -> exiD               | exid_new_loc  | all       |             6765 | 0.21 [0.16, 0.27] | 0.64 [0.60, 0.68]         |            0.25 | 0%           |
| shift: highD cal -> exiD               | exid_new_loc  | pre_drift |              590 | 0.20 [0.14, 0.26] | 0.25 [0.18, 0.34]         |            0.22 | 0%           |
| repair: exiD cal, highD-trained model  | exid_seen_loc | all       |             3866 | 0.81 [0.73, 0.87] | 0.89 [0.85, 0.92]         |            0.95 | 0%           |
| repair: exiD cal, highD-trained model  | exid_seen_loc | pre_drift |              503 | 0.84 [0.76, 0.91] | 0.68 [0.58, 0.76]         |            0.95 | 0%           |
| repair: exiD cal, highD-trained model  | exid_new_loc  | all       |             6765 | 0.52 [0.40, 0.64] | 0.90 [0.84, 0.94]         |            0.89 | 0%           |
| repair: exiD cal, highD-trained model  | exid_new_loc  | pre_drift |              590 | 0.51 [0.39, 0.62] | 0.64 [0.51, 0.78]         |            0.88 | 0%           |
| repair: exiD cal, highD+1% exiD model  | exid_seen_loc | all       |             3866 | 0.85 [0.79, 0.88] | 0.88 [0.85, 0.91]         |            0.9  | 0%           |
| repair: exiD cal, highD+1% exiD model  | exid_seen_loc | pre_drift |              503 | 0.89 [0.82, 0.92] | 0.51 [0.42, 0.59]         |            0.91 | 0%           |
| repair: exiD cal, highD+1% exiD model  | exid_new_loc  | all       |             6765 | 0.74 [0.63, 0.81] | 0.84 [0.81, 0.87]         |            0.8  | 0%           |
| repair: exiD cal, highD+1% exiD model  | exid_new_loc  | pre_drift |              590 | 0.75 [0.65, 0.82] | 0.35 [0.28, 0.44]         |            0.8  | 0%           |
| repair: exiD cal, highD+5% exiD model  | exid_seen_loc | all       |             3866 | 0.84 [0.78, 0.88] | 0.88 [0.84, 0.91]         |            0.88 | 0%           |
| repair: exiD cal, highD+5% exiD model  | exid_seen_loc | pre_drift |              503 | 0.88 [0.82, 0.92] | 0.43 [0.32, 0.50]         |            0.9  | 0%           |
| repair: exiD cal, highD+5% exiD model  | exid_new_loc  | all       |             6765 | 0.78 [0.64, 0.86] | 0.82 [0.78, 0.84]         |            0.81 | 0%           |
| repair: exiD cal, highD+5% exiD model  | exid_new_loc  | pre_drift |              590 | 0.79 [0.67, 0.87] | 0.21 [0.16, 0.26]         |            0.81 | 0%           |
| repair: exiD cal, highD+20% exiD model | exid_seen_loc | all       |             3866 | 0.85 [0.77, 0.88] | 0.89 [0.86, 0.91]         |            0.88 | 0%           |
| repair: exiD cal, highD+20% exiD model | exid_seen_loc | pre_drift |              503 | 0.89 [0.82, 0.92] | 0.54 [0.45, 0.60]         |            0.9  | 0%           |
| repair: exiD cal, highD+20% exiD model | exid_new_loc  | all       |             6765 | 0.79 [0.69, 0.85] | 0.81 [0.78, 0.83]         |            0.81 | 0%           |
| repair: exiD cal, highD+20% exiD model | exid_new_loc  | pre_drift |              590 | 0.81 [0.70, 0.87] | 0.22 [0.17, 0.29]         |            0.82 | 0%           |
| exiD-only model, exiD cal              | exid_seen_loc | all       |             3866 | 0.85 [0.80, 0.88] | 0.89 [0.88, 0.91]         |            0.87 | 0%           |
| exiD-only model, exiD cal              | exid_seen_loc | pre_drift |              503 | 0.90 [0.85, 0.93] | 0.55 [0.48, 0.61]         |            0.9  | 0%           |
| exiD-only model, exiD cal              | exid_new_loc  | all       |             6765 | 0.71 [0.62, 0.79] | 0.87 [0.84, 0.89]         |            0.75 | 0%           |
| exiD-only model, exiD cal              | exid_new_loc  | pre_drift |              590 | 0.72 [0.62, 0.80] | 0.36 [0.29, 0.45]         |            0.75 | 0%           |

## Scenario mining, precision@k [query-draw 95% interval]

| scenario                                 | tag           |   prevalence |   pool_pos | cnn               | random-init CNN (control)   | raw BEV (PCA-64)   | scalar features   |
|:-----------------------------------------|:--------------|-------------:|-----------:|:------------------|:----------------------------|:-------------------|:------------------|
| exiD -> exiD (exiD-trained)              | lane_change   |        0.05  |       6765 | 0.90 [0.73, 0.96] | 0.97 [0.90, 1.00]           | 0.04 [0.00, 0.17]  | 0.98 [0.95, 1.00] |
| exiD -> exiD (exiD-trained)              | ttc_crit_next |        0.003 |        413 | 0.04 [0.00, 0.10] | 0.09 [0.03, 0.16]           | 0.08 [0.03, 0.14]  | 0.27 [0.14, 0.36] |
| highD -> exiD (highD-trained, zero-shot) | lane_change   |        0.064 |       3866 | 0.86 [0.77, 0.94] | 0.97 [0.94, 0.99]           | 0.13 [0.02, 0.25]  | 0.98 [0.97, 1.00] |
| highD -> exiD (highD-trained, zero-shot) | ttc_crit_next |        0.004 |        217 | 0.09 [0.05, 0.13] | 0.07 [0.04, 0.09]           | 0.02 [0.00, 0.04]  | 0.23 [0.18, 0.30] |
| highD -> highD                           | lane_change   |        0.02  |        943 | 0.99 [0.98, 1.00] | 1.00 [0.98, 1.00]           | 0.13 [0.00, 0.33]  | 1.00 [0.99, 1.00] |

## Energy (modelled)

Tracks: 140180. Modelled mean 200.51 Wh/km. By dataset: {'exid': {'mean': 207.32, 'std': 167.07, 'count': 51728}, 'highd': {'mean': 196.53, 'std': 144.42, 'count': 88452}}.

Clusters (per k): {"2": {"silhouette": 0.47, "ari_mean": 0.978, "ari_median": 0.981, "ari_p10": 0.968, "ari_min": 0.95}, "3": {"silhouette": 0.467, "ari_mean": 0.557, "ari_median": 0.456, "ari_p10": 0.432, "ari_min": 0.42}, "4": {"silhouette": 0.251, "ari_mean": 0.824, "ari_median": 0.772, "ari_p10": 0.753, "ari_min": 0.736}, "5": {"silhouette": 0.28, "ari_mean": 0.706, "ari_median": 0.544, "ari_p10": 0.514, "ari_min": 0.487}}

Chosen k = 2, stability criterion met: True.

|   cluster |      n |   wh_km |   mean_speed |   speed_std |   p95_abs_acc |   jerk_rms |   lane_changes |   mean_thw |
|----------:|-------:|--------:|-------------:|------------:|--------------:|-----------:|---------------:|-----------:|
|         0 | 118055 |  199.73 |        31.34 |        0.72 |          0.42 |       0.4  |           0.1  |       2.54 |
|         1 |  22125 |  204.63 |        23.62 |        2.7  |          1.21 |       0.67 |           0.69 |       2.55 |

Style beyond speed: {"r2_speed_only": 0.046, "r2_speed_plus_cluster": 0.0559, "delta_r2": 0.0099, "binned_speed_control": {"r2_without_cluster": 0.0529, "r2_with_cluster": 0.0583, "delta_r2": 0.0054, "cluster_effect_wh_km": {"1": 36.62}}, "delta_r2_ci": [0.0028, 0.0206], "cluster_effect_wh_km_vs_cluster0": {"1": [48.3, 26.6, 70.17]}}

Acceleration source: {"smoothed": 200.51, "dataset_provided": 199.8, "track_corr": 1.0}

Largest sensitivities (+-20% on one parameter):

| variant       |   mean_wh_km |   change_pct |
|:--------------|-------------:|-------------:|
| eta_drive-20% |       252.98 |        26.17 |
| eta_drive+20% |       165.52 |       -17.45 |
| cda_m2+20%    |       222.16 |        10.8  |
| rho_air+20%   |       222.16 |        10.8  |
| rho_air-20%   |       179.11 |       -10.67 |
| cda_m2-20%    |       179.11 |       -10.67 |
| mass_kg+20%   |       218.7  |         9.07 |
| mass_kg-20%   |       182.82 |        -8.82 |

## Tuning and reproducibility

Boosting (validation AP at 2 s):

|   max_leaf_nodes |   learning_rate |   val_ap_src |   val_ap_exid |   val_ap_mean |
|-----------------:|----------------:|-------------:|--------------:|--------------:|
|               63 |            0.1  |       0.9817 |        0.8855 |        0.9336 |
|               31 |            0.1  |       0.9802 |        0.8828 |        0.9315 |
|               63 |            0.06 |       0.9793 |        0.8835 |        0.9314 |
|               63 |            0.03 |       0.9796 |        0.8799 |        0.9298 |
|               31 |            0.06 |       0.9796 |        0.8784 |        0.929  |
|               15 |            0.1  |       0.9787 |        0.8729 |        0.9258 |
|               15 |            0.06 |       0.9787 |        0.871  |        0.9248 |
|               31 |            0.03 |       0.9738 |        0.8718 |        0.9228 |
|               15 |            0.03 |       0.9794 |        0.8609 |        0.9201 |

CNN (validation PR-AUC averaged over horizons):

| override                  |   exid |    src |   mean |
|:--------------------------|-------:|-------:|-------:|
| {"lr": 0.004}             | 0.8572 | 0.9562 | 0.9067 |
| {}                        | 0.853  | 0.9508 | 0.9019 |
| {"weight_decay": 0.001}   | 0.8504 | 0.9502 | 0.9003 |
| {"widths": [48, 96, 128]} | 0.8502 | 0.9502 | 0.9002 |
| {"widths": [16, 32, 48]}  | 0.8477 | 0.9515 | 0.8996 |
| {"lr": 0.001}             | 0.8449 | 0.9427 | 0.8938 |

Reproducibility check: {"highd_7": {"ingest_equal": true, "labels_equal": true, "features_max_abs_diff": 0.0, "bev_equal": true}, "highd_41": {"ingest_equal": true, "labels_equal": true, "features_max_abs_diff": 0.0, "bev_equal": true}, "exid_12": {"ingest_equal": true, "labels_equal": true, "features_max_abs_diff": 0.0, "bev_equal": true}, "exid_70": {"ingest_equal": true, "labels_equal": true, "features_max_abs_diff": 0.0, "bev_equal": true}, "hgb_deterministic_max_abs_diff": 0.0, "cnn_cpu_deterministic_max_abs_diff": 0.0}