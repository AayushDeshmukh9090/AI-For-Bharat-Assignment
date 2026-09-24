## Ablations (eval = official valid, n per run below)

| run           |   WER raw |   CER raw |   WER norm |   CER norm |   CER no-space |   WER read |   WER spont. | trainable M   | train min   | peak GB   |
|:--------------|----------:|----------:|-----------:|-----------:|---------------:|-----------:|-------------:|:--------------|:------------|:----------|
| base          |     12.33 |      4.24 |      12.33 |       4.24 |           4.39 |       5.95 |        12.81 | -             | -           | -         |
| A1_full       |     13.11 |      4.56 |      13.11 |       4.56 |           4.71 |       7.19 |        13.55 | 1221          | 10.2        | 61.5      |
| A2_frozen_enc |     13.99 |      5.02 |      13.99 |       5.02 |           5.15 |       8.2  |        14.42 | 410           | 11.1        | 17.7      |
| A3_half_data  |     13.38 |      4.68 |      13.38 |       4.68 |           4.8  |       8.01 |        13.78 | 1221          | 7.8         | 57.3      |
| A4_large      |     12.83 |      4.42 |      12.83 |       4.42 |           4.58 |       7.83 |        13.2  | 1221          | 26.6        | 54.9      |
| A5_top2       |     14.37 |      5.06 |      14.37 |       5.06 |           5.14 |       8.35 |        14.82 | 84            | 13.8        | 12.8      |

## Subgroups: base vs A4_large (normalized WER)

| axis       | group        |    n |   WER base |   WER A4_large |   ΔWER |
|:-----------|:-------------|-----:|-----------:|---------------:|-------:|
| register   | read         |  204 |       5.95 |           7.83 |   1.88 |
| register   | spontaneous  | 3351 |      12.81 |          13.2  |   0.39 |
| scenario   | Conversation | 2078 |      16.07 |          16.38 |   0.31 |
| scenario   | Extempore    | 1273 |       9.85 |          10.32 |   0.47 |
| scenario   | Read         |  204 |       5.95 |           7.83 |   1.88 |
| dur_bucket | long >10s    |  767 |      11.06 |          11.63 |   0.57 |
| dur_bucket | medium 2-10s | 1679 |      12.75 |          13.23 |   0.48 |
| dur_bucket | short <2s    | 1109 |      19.7  |          19.84 |   0.14 |
| gender     | Female       | 1846 |      11.33 |          11.86 |   0.53 |
| gender     | Male         | 1707 |      13.46 |          13.93 |   0.47 |
| gender     | Other        |    2 |       0    |           0    |   0    |
