# Results summary

Mean ± standard deviation over the seeds, in %. gap = acc aligned - acc conflicting (large = the model relies on the colour). shortcut rate = how often the prediction is the colour's class on the conflicting set.

## MLP vs CNN, clean vs biased training (p = 0.9)

| model | trained on | params | acc aligned | acc clean (no colour) | acc random colour | acc conflicting | shortcut rate | gap        | seeds |
|-------|------------|--------|-------------|-----------------------|-------------------|-----------------|---------------|------------|-------|
| MLP   | clean      | 103982 | 95.2 ± 0.6  | 96.4 ± 0.2            | 96.0 ± 0.1        | 96.3 ± 0.2      | 0.3 ± 0.1     | -1.1 ± 0.5 | 3     |
| MLP   | biased     | 103982 | 99.0 ± 0.0  | 86.7 ± 1.1            | 84.4 ± 0.1        | 81.6 ± 1.3      | 6.4 ± 0.6     | 17.4 ± 1.3 | 3     |
| CNN   | clean      | 106154 | 96.3 ± 0.4  | 98.6 ± 0.1            | 97.3 ± 0.1        | 97.2 ± 0.9      | 0.2 ± 0.1     | -1.0 ± 1.3 | 3     |
| CNN   | biased     | 106154 | 99.6 ± 0.1  | 93.1 ± 1.8            | 93.0 ± 1.1        | 93.2 ± 0.3      | 2.7 ± 0.3     | 6.4 ± 0.2  | 3     |

## Mitigations (SmallCNN)

| p    | mitigation   | acc aligned | acc clean (no colour) | acc random colour | acc conflicting | shortcut rate | gap        | seeds |
|------|--------------|-------------|-----------------------|-------------------|-----------------|---------------|------------|-------|
| 0.5  | none         | 99.1 ± 0.1  | 97.6 ± 0.2            | 97.7 ± 0.2        | 97.3 ± 0.1      | 0.6 ± 0.1     | 1.8 ± 0.1  | 3     |
| 0.9  | none         | 99.6 ± 0.1  | 93.1 ± 1.8            | 93.0 ± 1.1        | 93.2 ± 0.3      | 2.7 ± 0.3     | 6.4 ± 0.2  | 3     |
| 0.9  | augmentation | 98.3 ± 0.2  | 98.3 ± 0.2            | 98.3 ± 0.2        | 98.3 ± 0.2      | 0.2 ± 0.1     | -0.0 ± 0.0 | 3     |
| 0.9  | rebalancing  | 99.3 ± 0.1  | 98.5 ± 0.1            | 98.4 ± 0.2        | 98.0 ± 0.2      | 0.3 ± 0.1     | 1.2 ± 0.1  | 3     |
| 0.9  | reweighting  | 98.4 ± 0.2  | 96.6 ± 0.2            | 96.1 ± 0.2        | 95.5 ± 0.3      | 0.7 ± 0.2     | 2.9 ± 0.4  | 3     |
| 0.99 | none         | 99.9 ± 0.0  | 67.2 ± 5.7            | 62.5 ± 3.8        | 58.0 ± 4.6      | 28.3 ± 5.7    | 41.9 ± 4.6 | 3     |
| 0.99 | augmentation | 98.3 ± 0.2  | 98.3 ± 0.2            | 98.3 ± 0.2        | 98.3 ± 0.2      | 0.2 ± 0.1     | -0.0 ± 0.0 | 3     |
| 0.99 | rebalancing  | 99.2 ± 0.1  | 98.5 ± 0.1            | 98.5 ± 0.1        | 98.3 ± 0.2      | 0.2 ± 0.1     | 1.0 ± 0.1  | 3     |
| 0.99 | reweighting  | 99.4 ± 0.0  | 83.6 ± 5.1            | 83.4 ± 1.4        | 80.8 ± 1.3      | 9.9 ± 1.8     | 18.6 ± 1.3 | 3     |
