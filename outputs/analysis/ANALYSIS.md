# M3 analyses - UPMC Food-101 (complete test pairs)

Mean ± std over the seeds available (percent). Produced by `scripts/analysis_m3.py`.

## Support / conflict over seeds

Groups from the unimodal models of the same seed: support = UA and UB right; conflict_I = only the image model right; conflict_T = only the text model right. 'Follows image' = the fused prediction equals UA's.

| group | items | share % | F0 late acc. | F1 concat acc. | F2 cross-modal acc. | F0 late follows image | F1 concat follows image | F2 cross-modal follows image |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| support | 13938 ± 133 | 61.7 ± 0.6 | 100.0 ± 0.0 | 99.8 ± 0.0 | 99.8 ± 0.0 | 100.0 ± 0.0 | 99.8 ± 0.0 | 99.8 ± 0.0 |
| conflict_I | 2043 ± 153 | 9.0 ± 0.7 | 90.6 ± 4.5 | 69.0 ± 2.7 | 75.3 ± 2.0 | 90.6 ± 4.5 | 69.0 ± 2.7 | 75.3 ± 2.0 |
| conflict_T | 5588 ± 74 | 24.7 ± 0.3 | 97.1 ± 0.9 | 97.7 ± 0.1 | 97.7 ± 0.2 | 2.9 ± 0.9 | 1.1 ± 0.2 | 1.2 ± 0.2 |
| both_wrong_agree | 150 ± 11 | 0.7 ± 0.0 | 0.0 ± 0.0 | 10.7 ± 3.5 | 10.6 ± 4.5 | 100.0 ± 0.0 | 83.3 ± 5.1 | 84.3 ± 6.6 |
| both_wrong_disagree | 875 ± 60 | 3.9 ± 0.3 | 0.7 ± 0.5 | 21.3 ± 4.3 | 24.0 ± 6.3 | 71.2 ± 7.1 | 15.7 ± 1.5 | 18.9 ± 1.5 |

F2 ablations on the same groups:

| group | items | share % | F2 no aux (E1) acc. | F2 no mod. dropout (E4) acc. | F2 no aux (E1) follows image | F2 no mod. dropout (E4) follows image |
| --- | --- | --- | --- | --- | --- | --- |
| support | 13938 ± 133 | 61.7 ± 0.6 | 99.8 ± 0.0 | 99.8 ± 0.0 | 99.8 ± 0.0 | 99.8 ± 0.0 |
| conflict_I | 2043 ± 153 | 9.0 ± 0.7 | 70.9 ± 1.3 | 74.7 ± 1.6 | 70.9 ± 1.3 | 74.7 ± 1.6 |
| conflict_T | 5588 ± 74 | 24.7 ± 0.3 | 97.9 ± 0.2 | 97.8 ± 0.2 | 1.0 ± 0.1 | 1.1 ± 0.1 |
| both_wrong_agree | 150 ± 11 | 0.7 ± 0.0 | 10.7 ± 3.4 | 10.0 ± 5.0 | 82.5 ± 7.9 | 84.0 ± 6.7 |
| both_wrong_disagree | 875 ± 60 | 3.9 ± 0.3 | 22.6 ± 4.4 | 23.2 ± 6.7 | 17.3 ± 1.2 | 18.9 ± 0.5 |

## Support / conflict with class names masked (E2: UA vs UB-mask)

| group | items | share % | F0-mask acc. | F1-mask acc. | F2-mask acc. | F0-mask follows image | F1-mask follows image | F2-mask follows image |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| support | 7044 ± 56 | 31.2 ± 0.2 | 100.0 ± 0.0 | 99.0 ± 0.1 | 99.3 ± 0.1 | 100.0 ± 0.0 | 99.0 ± 0.1 | 99.3 ± 0.1 |
| conflict_I | 8937 ± 46 | 39.6 ± 0.2 | 95.9 ± 1.8 | 87.4 ± 0.2 | 87.5 ± 0.1 | 95.9 ± 1.8 | 87.4 ± 0.2 | 87.5 ± 0.1 |
| conflict_T | 2812 ± 19 | 12.4 ± 0.1 | 75.9 ± 4.3 | 80.8 ± 0.2 | 85.7 ± 0.2 | 23.9 ± 4.4 | 7.7 ± 0.3 | 6.4 ± 0.1 |
| both_wrong_agree | 195 ± 11 | 0.9 ± 0.0 | 0.0 ± 0.0 | 15.3 ± 1.1 | 12.8 ± 0.3 | 100.0 ± 0.0 | 75.0 ± 2.3 | 78.7 ± 1.9 |
| both_wrong_disagree | 3606 ± 18 | 16.0 ± 0.1 | 1.7 ± 0.2 | 30.3 ± 0.8 | 31.2 ± 0.4 | 75.3 ± 4.2 | 22.2 ± 0.6 | 22.0 ± 0.1 |

## Macro-F1 with the title removed from a fraction of test pairs

| model | 0 % | 25 % | 50 % | 75 % | 100 % |
| --- | --- | --- | --- | --- | --- |
| UA image | 70.8 ± 0.1 | 70.8 ± 0.1 | 70.8 ± 0.1 | 70.8 ± 0.1 | 70.8 ± 0.1 |
| UB text | 86.8 ± 0.3 | 73.4 ± 0.3 | 57.3 ± 0.4 | 34.1 ± 0.4 | 0.0 ± 0.0 |
| F0 late | 93.9 ± 0.2 | 88.1 ± 0.2 | 82.2 ± 0.1 | 76.7 ± 0.1 | 70.8 ± 0.1 |
| F1 concat | 92.8 ± 0.1 | 82.9 ± 0.1 | 73.1 ± 0.2 | 63.0 ± 0.3 | 52.5 ± 0.4 |
| F2 cross-modal | 93.5 ± 0.1 | 86.6 ± 0.4 | 79.5 ± 0.5 | 72.5 ± 0.6 | 65.3 ± 0.9 |
| F2 no aux (E1) | 93.1 ± 0.1 | 84.5 ± 0.3 | 76.0 ± 0.4 | 67.3 ± 0.6 | 58.5 ± 0.8 |
| F2 no mod. dropout (E4) | 93.4 ± 0.1 | 85.3 ± 0.2 | 77.1 ± 0.3 | 68.8 ± 0.3 | 60.1 ± 0.6 |

## Macro-F1 with the image removed from a fraction of test pairs

| model | 0 % | 25 % | 50 % | 75 % | 100 % |
| --- | --- | --- | --- | --- | --- |
| UA image | 70.8 ± 0.1 | 60.0 ± 0.1 | 47.2 ± 0.0 | 28.0 ± 0.0 | 0.0 ± 0.0 |
| UB text | 86.8 ± 0.3 | 86.8 ± 0.3 | 86.8 ± 0.3 | 86.8 ± 0.3 | 86.8 ± 0.3 |
| F0 late | 93.9 ± 0.2 | 92.1 ± 0.3 | 90.4 ± 0.3 | 88.7 ± 0.3 | 86.8 ± 0.3 |
| F1 concat | 92.8 ± 0.1 | 91.3 ± 0.1 | 89.7 ± 0.1 | 88.1 ± 0.0 | 86.3 ± 0.1 |
| F2 cross-modal | 93.5 ± 0.1 | 91.9 ± 0.1 | 90.3 ± 0.1 | 88.6 ± 0.1 | 86.7 ± 0.1 |
| F2 no aux (E1) | 93.1 ± 0.1 | 91.5 ± 0.1 | 89.8 ± 0.0 | 88.2 ± 0.1 | 86.4 ± 0.1 |
| F2 no mod. dropout (E4) | 93.4 ± 0.1 | 91.8 ± 0.1 | 90.1 ± 0.1 | 88.4 ± 0.1 | 86.5 ± 0.0 |

## E1 - auxiliary unimodal losses (F2, λ = 0.5 vs 0)

With λ = 0 the unimodal heads are trained on detached features (linear probes of the encoders).

| model | macro-F1 | accuracy | image-head acc. | text-head acc. | acc. image removed | acc. text removed |
| --- | --- | --- | --- | --- | --- | --- |
| F2 cross-modal | 93.53 ± 0.07 | 93.58 ± 0.07 | 69.7 ± 0.4 | 87.1 ± 0.0 | 86.7 ± 0.2 | 65.3 ± 0.9 |
| F2 no aux (E1) | 93.07 ± 0.08 | 93.13 ± 0.08 | 62.3 ± 0.1 | 76.8 ± 1.7 | 86.3 ± 0.1 | 58.6 ± 0.8 |

## E4 - modality dropout (F2, p = 0.15 vs 0)

| model | macro-F1 | acc. text removed | acc. image removed | missing-text set, title as is | missing-text set, [MISSING] token |
| --- | --- | --- | --- | --- | --- |
| F2 cross-modal | 93.53 ± 0.07 | 65.3 ± 0.9 | 86.7 ± 0.2 | 61.4 ± 0.5 | 60.6 ± 2.4 |
| F2 no mod. dropout (E4) | 93.45 ± 0.06 | 60.6 ± 0.4 | 86.5 ± 0.0 | 58.1 ± 1.9 | 57.9 ± 1.4 |

## E2 - class names masked in every title (train, val and test)

| model | macro-F1 as is | macro-F1 masked | Δ (pp) | masked: acc. no class mention | masked: acc. text removed |
| --- | --- | --- | --- | --- | --- |
| T0 TF-IDF | 87.26 | 47.79 | -39.47 | 42.7 | - |
| UB text | 86.80 ± 0.25 | 43.70 ± 0.16 | -43.10 | 39.7 ± 0.2 | - |
| F0 late | 93.86 ± 0.24 | 78.85 ± 0.17 | -15.01 | 76.8 ± 0.0 | 70.7 ± 0.1 |
| F1 concat | 92.80 ± 0.06 | 80.43 ± 0.14 | -12.37 | 76.1 ± 0.2 | 66.6 ± 0.0 |
| F2 cross-modal | 93.53 ± 0.07 | 81.31 ± 0.02 | -12.22 | 76.6 ± 0.4 | 68.6 ± 0.1 |

Fusion gain over the text model of the same seed (macro-F1, pp):

| fusion | gain over UB (as is) | gain over UB-mask (masked) |
| --- | --- | --- |
| F0 late | 7.06 ± 0.15 | 35.15 ± 0.28 |
| F1 concat | 6.00 ± 0.31 | 36.73 ± 0.03 |
| F2 cross-modal | 6.73 ± 0.31 | 37.61 ± 0.16 |

## Calibration: test ECE before / after temperature scaling (T fitted on validation)

| model | T | ECE before | ECE after |
| --- | --- | --- | --- |
| UA image | 1.62 ± 0.03 | 10.75 ± 0.19 | 4.22 ± 0.49 |
| UB text | 1.30 ± 0.20 | 2.84 ± 1.46 | 1.01 ± 0.23 |
| F0 late | 0.61 ± 0.02 | 15.98 ± 0.78 | 6.46 ± 1.27 |
| F1 concat | 1.91 ± 0.03 | 4.69 ± 0.02 | 0.90 ± 0.14 |
| F2 cross-modal | 1.40 ± 0.02 | 2.88 ± 0.09 | 0.45 ± 0.04 |
| F2 no aux (E1) | 1.39 ± 0.05 | 3.02 ± 0.14 | 0.59 ± 0.06 |
| F2 no mod. dropout (E4) | 1.59 ± 0.00 | 3.57 ± 0.05 | 0.56 ± 0.03 |
| UB-mask | 1.30 ± 0.00 | 7.02 ± 0.06 | 3.16 ± 0.21 |
| F0-mask | 0.70 ± 0.01 | 23.45 ± 0.90 | 9.03 ± 0.74 |
| F1-mask | 1.91 ± 0.03 | 11.15 ± 0.19 | 0.81 ± 0.27 |
| F2-mask | 1.50 ± 0.00 | 8.43 ± 0.01 | 0.83 ± 0.11 |
| T0 TF-IDF | 0.87 | 2.71 | 1.09 |
| T0-mask | 1.22 | 5.47 | 4.81 |

## [FUS] attention on image tokens (last fusion layer, seed 42, mean share)

| model | all | support | conflict_I | conflict_T | both_wrong |
| --- | --- | --- | --- | --- | --- |
| F2 cross-modal | 0.805 | 0.792 | 0.909 | 0.778 | 0.891 |
| F2-mask | 0.922 | 0.928 | 0.927 | 0.907 | 0.912 |
| F2 no aux (E1) | 0.756 | 0.745 | 0.873 | 0.720 | 0.847 |
| F2 no mod. dropout (E4) | 0.883 | 0.883 | 0.930 | 0.858 | 0.917 |
