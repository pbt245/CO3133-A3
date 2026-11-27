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
