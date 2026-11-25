# M2 comparison - UPMC Food-101 (complete test pairs)

Mean ± std over seeds (percent). Test = complete test pairs (valid image + valid title). Dedup = without test items linked to an official-train image and without cross-label clusters.

| model | seeds | accuracy | macro-F1 | top-5 | ECE | dedup macro-F1 | val macro-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| K0 keyword rule | 42 | 76.95 | 83.56 | 77.41 | 2.97 | 83.81 | 84.16 |
| UA image (ResNet-50) | 42 123 2026 | 70.73 ± 0.09 | 70.80 ± 0.06 | 86.51 ± 0.16 | 10.75 ± 0.19 | 67.72 ± 0.03 | 68.22 ± 0.44 |
| UB text (DistilBERT) | 42 123 2026 | 86.42 ± 0.91 | 86.80 ± 0.25 | 90.71 ± 0.69 | 2.84 ± 1.46 | 86.96 ± 0.07 | 87.17 ± 0.11 |
| F0 late fusion | 42 123 2026 | 93.93 ± 0.23 | 93.86 ± 0.24 | 97.71 ± 0.07 | 15.98 ± 0.78 | 93.71 ± 0.27 | 94.06 ± 0.26 |
| F1 concat fusion | 42 123 2026 | 92.85 ± 0.06 | 92.80 ± 0.06 | 97.15 ± 0.04 | 4.69 ± 0.02 | 92.80 ± 0.07 | 93.05 ± 0.07 |

Test complete pairs: 22,594; dedup subset: 19,270; titles without a class mention: 4,782; natural missing-text set: 121.

## Where the text helps: accuracy by subset

| model | title mentions class | no class mention | missing-text set (as is) | missing-text, text zero-filled / fallback | image removed | text removed |
| --- | --- | --- | --- | --- | --- | --- |
| K0 keyword rule | 97.48 | 0.46 | 0.00 | - | - | - |
| UA image (ResNet-50) | 71.74 ± 0.07 | 66.96 ± 0.17 | 65.56 ± 1.91 | - | - | - |
| UB text (DistilBERT) | 99.10 ± 0.07 | 39.22 ± 4.07 | 6.34 ± 1.72 | - | - | - |
| F0 late fusion | 99.03 ± 0.16 | 74.95 ± 1.58 | 68.87 ± 2.66 | 65.56 ± 1.91 | 86.42 ± 0.91 | 70.73 ± 0.09 |
| F1 concat fusion | 99.05 ± 0.02 | 69.76 ± 0.24 | 51.79 ± 1.72 | 51.79 ± 1.72 | 86.27 ± 0.06 | 53.32 ± 0.32 |

## Cost

| model | params (M) | train time (min) | epochs run / best | throughput (items/s, bs 128) | latency (ms, bs 1) |
| --- | --- | --- | --- | --- | --- |
| K0 keyword rule | 0.0 | 0.0 | - | - | - |
| UA image (ResNet-50) | 23.7 | 14.4 | 10.0 / 9.0 | 2793 | 4.68 |
| UB text (DistilBERT) | 66.4 | 2.0 | 4.7 / 3.3 | 17105 | 2.69 |
| F0 late fusion | 90.2 | 16.4 | - | - | - |
| F1 concat fusion | 91.9 | 18.4 | 10.0 / 9.0 | 2385 | 7.71 |

F0 train time = UA + UB; F0 weight on the image (tuned on val): seed 123: 0.45, seed 2026: 0.50, seed 42: 0.45

## Decision rule (mean macro-F1 difference > larger std AND McNemar p < 0.05, seed 42)

| A vs B | Δ macro-F1 (pp) | larger std (pp) | McNemar b / c, p | verdict |
| --- | --- | --- | --- | --- |
| UB text (DistilBERT) vs K0 keyword rule | +3.24 | 0.25 | 2031 / 128, 0 | ub_distilbert better |
| UB text (DistilBERT) vs UA image (ResNet-50) | +16.00 | 0.25 | 5503 / 2219, 1.7e-305 | ub_distilbert better |
| F0 late fusion vs UB text (DistilBERT) | +7.06 | 0.25 | 2068 / 178, 0 | f0_late_fusion better |
| F1 concat fusion vs UB text (DistilBERT) | +6.00 | 0.25 | 1868 / 161, 0 | f1_concat better |
| F1 concat fusion vs UA image (ResNet-50) | +22.00 | 0.06 | 5641 / 650, 0 | f1_concat better |
| F1 concat fusion vs F0 late fusion | -1.06 | 0.24 | 426 / 609, 1.5e-08 | f0_late_fusion better |

## Title leakage check: titles seen / unseen in train

8,893 of 22,594 complete test pairs (39.4 %) have a normalised title that also occurs in train/val (several images of one web page, or a generic title). The image-duplicate grouping does not see these links.

| model | acc. title seen | acc. title unseen | macro-F1 title unseen |
| --- | --- | --- | --- |
| K0 keyword rule | 76.14 | 77.47 | 82.89 |
| UA image (ResNet-50) | 70.18 ± 0.23 | 71.09 ± 0.10 | 70.59 ± 0.12 |
| UB text (DistilBERT) | 91.71 ± 1.81 | 82.99 ± 0.34 | 82.57 ± 0.44 |
| F0 late fusion | 95.88 ± 0.50 | 92.67 ± 0.30 | 92.03 ± 0.32 |
| F1 concat fusion | 95.76 ± 0.16 | 90.96 ± 0.02 | 90.28 ± 0.03 |

## Support / conflict (preliminary, seed 42)

Groups by unimodal correctness: support = UA and UB right; conflict_I = only the image model right; conflict_T = only the text model right.

| group | items | share | f0_late_fusion acc. | f1_concat acc. | f0_late_fusion follows image | f1_concat follows image |
| --- | --- | --- | --- | --- | --- | --- |
| support | 13,785 | 61.0 % | 100.0 | 99.8 | 100.0 | 99.8 |
| conflict_I | 2,219 | 9.8 % | 92.7 | 72.1 | 92.7 | 72.1 |
| conflict_T | 5,503 | 24.4 % | 96.8 | 97.6 | 3.2 | 1.3 |
| both_wrong_agree | 143 | 0.6 % | 0.0 | 14.0 | 100.0 | 77.6 |
| both_wrong_disagree | 944 | 4.2 % | 1.3 | 26.3 | 78.0 | 15.6 |

The same table split by class-name mention and by title seen / unseen is in support_conflict.csv.
