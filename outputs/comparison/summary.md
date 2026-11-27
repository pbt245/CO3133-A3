# Comparison - UPMC Food-101 (complete test pairs)

Mean ± std over seeds (percent). Test = complete test pairs (valid image + valid title). Dedup = without test items linked to an official-train image and without cross-label clusters.

| model | seeds | accuracy | macro-F1 | top-5 | ECE | dedup macro-F1 | val macro-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| K0 keyword rule | 42 | 76.95 | 83.56 | 77.41 | 2.97 | 83.81 | 84.16 |
| T0 TF-IDF + LR | 42 | 87.12 | 87.26 | 91.23 | 2.71 | 87.36 | 87.90 |
| UA image (ResNet-50) | 42 123 2026 | 70.73 ± 0.09 | 70.80 ± 0.06 | 86.51 ± 0.16 | 10.75 ± 0.19 | 67.72 ± 0.03 | 68.22 ± 0.44 |
| UB text (DistilBERT) | 42 123 2026 | 86.42 ± 0.91 | 86.80 ± 0.25 | 90.71 ± 0.69 | 2.84 ± 1.46 | 86.96 ± 0.07 | 87.17 ± 0.11 |
| F0 late fusion | 42 123 2026 | 93.93 ± 0.23 | 93.86 ± 0.24 | 97.71 ± 0.07 | 15.98 ± 0.78 | 93.71 ± 0.27 | 94.06 ± 0.26 |
| F1 concat fusion | 42 123 2026 | 92.85 ± 0.06 | 92.80 ± 0.06 | 97.15 ± 0.04 | 4.69 ± 0.02 | 92.80 ± 0.07 | 93.05 ± 0.07 |
| F2 cross-modal fusion | 42 123 2026 | 93.58 ± 0.07 | 93.53 ± 0.07 | 97.69 ± 0.04 | 2.88 ± 0.09 | 93.49 ± 0.06 | 93.95 ± 0.03 |
| E1: F2 without aux. losses | 42 123 2026 | 93.13 ± 0.08 | 93.07 ± 0.08 | 97.33 ± 0.09 | 3.01 ± 0.14 | 92.99 ± 0.07 | 93.48 ± 0.27 |
| E4: F2 without modality dropout | 42 123 2026 | 93.50 ± 0.06 | 93.45 ± 0.06 | 97.45 ± 0.08 | 3.57 ± 0.05 | 93.41 ± 0.07 | 93.62 ± 0.19 |
| E2: T0, class names masked | 42 | 47.65 | 47.79 | 64.30 | 5.47 | 46.02 | 47.27 |
| E2: UB, class names masked | 42 123 2026 | 43.62 ± 0.16 | 43.70 ± 0.16 | 63.51 ± 0.19 | 7.02 ± 0.06 | 42.31 ± 0.17 | 44.70 ± 0.47 |
| E2: F0, class names masked | 42 123 2026 | 78.84 ± 0.15 | 78.85 ± 0.17 | 91.48 ± 0.09 | 23.45 ± 0.90 | 76.75 ± 0.18 | 78.38 ± 0.28 |
| E2: F1, class names masked | 42 123 2026 | 80.46 ± 0.15 | 80.43 ± 0.14 | 92.86 ± 0.22 | 11.14 ± 0.19 | 78.91 ± 0.12 | 80.40 ± 0.56 |
| E2: F2, class names masked | 42 123 2026 | 81.34 ± 0.02 | 81.31 ± 0.02 | 93.34 ± 0.13 | 8.43 ± 0.01 | 79.80 ± 0.06 | 81.22 ± 0.09 |

Test complete pairs: 22,594; dedup subset: 19,270; titles without a class mention: 4,782; natural missing-text set: 121.

## Where the text helps: accuracy by subset

| model | title mentions class | no class mention | missing-text set (as is) | missing-text, text zero-filled / fallback | image removed | text removed |
| --- | --- | --- | --- | --- | --- | --- |
| K0 keyword rule | 97.48 | 0.46 | 0.00 | - | - | - |
| T0 TF-IDF + LR | 99.12 | 42.39 | 8.26 | - | - | - |
| UA image (ResNet-50) | 71.74 ± 0.07 | 66.96 ± 0.17 | 65.56 ± 1.91 | - | - | - |
| UB text (DistilBERT) | 99.10 ± 0.07 | 39.22 ± 4.07 | 6.34 ± 1.72 | - | - | - |
| F0 late fusion | 99.03 ± 0.16 | 74.95 ± 1.58 | 68.87 ± 2.66 | 65.56 ± 1.91 | 86.42 ± 0.91 | 70.73 ± 0.09 |
| F1 concat fusion | 99.05 ± 0.02 | 69.76 ± 0.24 | 51.79 ± 1.72 | 51.79 ± 1.72 | 86.27 ± 0.06 | 53.32 ± 0.32 |
| F2 cross-modal fusion | 99.10 ± 0.02 | 73.02 ± 0.32 | 61.43 ± 0.48 | 60.61 ± 2.39 | 86.73 ± 0.16 | 65.30 ± 0.85 |
| E1: F2 without aux. losses | 99.12 ± 0.01 | 70.79 ± 0.36 | 53.72 ± 2.98 | 52.07 ± 2.98 | 86.32 ± 0.14 | 58.61 ± 0.84 |
| E4: F2 without modality dropout | 99.07 ± 0.04 | 72.74 ± 0.23 | 58.13 ± 1.91 | 57.85 ± 1.43 | 86.49 ± 0.03 | 60.63 ± 0.37 |
| E2: T0, class names masked | 48.97 | 42.72 | 8.26 | - | - | - |
| E2: UB, class names masked | 44.67 ± 0.21 | 39.70 ± 0.18 | 5.79 ± 1.43 | - | - | - |
| E2: F0, class names masked | 79.39 ± 0.19 | 76.82 ± 0.01 | 69.15 ± 1.72 | 65.56 ± 1.91 | 43.62 ± 0.16 | 70.73 ± 0.09 |
| E2: F1, class names masked | 81.64 ± 0.23 | 76.08 ± 0.18 | 61.71 ± 0.48 | 62.26 ± 2.90 | 36.30 ± 0.07 | 66.55 ± 0.03 |
| E2: F2, class names masked | 82.61 ± 0.14 | 76.60 ± 0.42 | 63.36 ± 2.90 | 64.19 ± 0.95 | 43.98 ± 0.17 | 68.64 ± 0.06 |

## Cost

| model | params (M) | train time (min) | epochs run / best | throughput (items/s, bs 128) | latency (ms, bs 1) |
| --- | --- | --- | --- | --- | --- |
| K0 keyword rule | 0.0 | 0.0 | - | - | - |
| T0 TF-IDF + LR | 13.6 | 4.4 | - | 241553 | - |
| UA image (ResNet-50) | 23.7 | 14.4 | 10.0 / 9.0 | 2793 | 4.68 |
| UB text (DistilBERT) | 66.4 | 2.0 | 4.7 / 3.3 | 17105 | 2.69 |
| F0 late fusion | 90.2 | 16.4 | - | - | - |
| F1 concat fusion | 91.9 | 18.4 | 10.0 / 9.0 | 2385 | 7.71 |
| F2 cross-modal fusion | 98.0 | 18.6 | 10.0 / 8.7 | 2220 | 8.50 |
| E1: F2 without aux. losses | 98.0 | 18.6 | 10.0 / 9.0 | 2214 | 8.76 |
| E4: F2 without modality dropout | 98.0 | 18.6 | 10.0 / 9.7 | 2218 | 8.51 |
| E2: T0, class names masked | 12.9 | 7.8 | - | 243236 | - |
| E2: UB, class names masked | 66.4 | 2.2 | 5.0 / 5.0 | 17341 | 2.57 |
| E2: F0, class names masked | 90.2 | 16.6 | - | - | - |
| E2: F1, class names masked | 91.9 | 18.3 | 10.0 / 10.0 | 2399 | 7.45 |
| E2: F2, class names masked | 98.0 | 18.6 | 10.0 / 10.0 | 2221 | 8.81 |

F0 train time = UA + UB; F0 weight on the image (tuned on val): seed 123: 0.45, seed 2026: 0.50, seed 42: 0.45

## Decision rule (mean macro-F1 difference > larger std AND McNemar p < 0.05, seed 42)

| A vs B | Δ macro-F1 (pp) | larger std (pp) | McNemar b / c, p | verdict |
| --- | --- | --- | --- | --- |
| UB text (DistilBERT) vs K0 keyword rule | +3.24 | 0.25 | 2031 / 128, 0 | ub_distilbert better |
| UB text (DistilBERT) vs T0 TF-IDF + LR | -0.46 | 0.25 | 260 / 655, 8.8e-39 | t0_tfidf better |
| UB text (DistilBERT) vs UA image (ResNet-50) | +16.00 | 0.25 | 5503 / 2219, 1.7e-305 | ub_distilbert better |
| F0 late fusion vs UB text (DistilBERT) | +7.06 | 0.25 | 2068 / 178, 0 | f0_late_fusion better |
| F1 concat fusion vs UB text (DistilBERT) | +6.00 | 0.25 | 1868 / 161, 0 | f1_concat better |
| F1 concat fusion vs UA image (ResNet-50) | +22.00 | 0.06 | 5641 / 650, 0 | f1_concat better |
| F1 concat fusion vs F0 late fusion | -1.06 | 0.24 | 426 / 609, 1.5e-08 | f0_late_fusion better |
| F2 cross-modal fusion vs UB text (DistilBERT) | +6.73 | 0.25 | 2036 / 168, 0 | f2_crossmodal better |
| F2 cross-modal fusion vs UA image (ResNet-50) | +22.73 | 0.07 | 5677 / 525, 0 | f2_crossmodal better |
| F2 cross-modal fusion vs F1 concat fusion | +0.73 | 0.07 | 439 / 278, 2.3e-09 | f2_crossmodal better |
| F2 cross-modal fusion vs F0 late fusion | -0.33 | 0.24 | 465 / 487, 0.5 | no decision (rule not met) |
| F2 cross-modal fusion vs E1: F2 without aux. losses | +0.46 | 0.08 | 379 / 250, 3.3e-07 | f2_crossmodal better |
| F2 cross-modal fusion vs E4: F2 without modality dropout | +0.08 | 0.07 | 263 / 248, 0.54 | no decision (rule not met) |
| UB text (DistilBERT) vs E2: UB, class names masked | +43.10 | 0.25 | 9902 / 491, 0 | ub_distilbert better |
| F1 concat fusion vs E2: F1, class names masked | +12.37 | 0.14 | 3330 / 531, 0 | f1_concat better |
| F2 cross-modal fusion vs E2: F2, class names masked | +12.22 | 0.07 | 3201 / 418, 0 | f2_crossmodal better |
| E2: UB, class names masked vs UA image (ResNet-50) | -27.10 | 0.16 | 2800 / 8927, 0 | ua_resnet50 better |
| E2: F0, class names masked vs E2: UB, class names masked | +35.15 | 0.17 | 8435 / 537, 0 | f0_late_fusion_mask better |
| E2: F1, class names masked vs E2: UB, class names masked | +36.73 | 0.16 | 8938 / 619, 0 | f1_concat_mask better |
| E2: F2, class names masked vs E2: UB, class names masked | +37.61 | 0.16 | 8950 / 454, 0 | f2_crossmodal_mask better |
| E2: F2, class names masked vs E2: F1, class names masked | +0.88 | 0.14 | 1202 / 1025, 0.00019 | f2_crossmodal_mask better |
| E2: F2, class names masked vs E2: F0, class names masked | +2.46 | 0.17 | 1760 / 1162, 2.3e-28 | f2_crossmodal_mask better |

## Title leakage check: titles seen / unseen in train

8,893 of 22,594 complete test pairs (39.4 %) have a normalised title that also occurs in train/val (several images of one web page, or a generic title). The image-duplicate grouping does not see these links.

| model | acc. title seen | acc. title unseen | macro-F1 title unseen |
| --- | --- | --- | --- |
| K0 keyword rule | 76.14 | 77.47 | 82.89 |
| T0 TF-IDF + LR | 93.24 | 83.14 | 82.57 |
| UA image (ResNet-50) | 70.18 ± 0.23 | 71.09 ± 0.10 | 70.59 ± 0.12 |
| UB text (DistilBERT) | 91.71 ± 1.81 | 82.99 ± 0.34 | 82.57 ± 0.44 |
| F0 late fusion | 95.88 ± 0.50 | 92.67 ± 0.30 | 92.03 ± 0.32 |
| F1 concat fusion | 95.76 ± 0.16 | 90.96 ± 0.02 | 90.28 ± 0.03 |
| F2 cross-modal fusion | 96.44 ± 0.06 | 91.73 ± 0.11 | 91.05 ± 0.14 |
| E1: F2 without aux. losses | 96.08 ± 0.08 | 91.21 ± 0.08 | 90.52 ± 0.12 |
| E4: F2 without modality dropout | 96.32 ± 0.02 | 91.67 ± 0.09 | 90.98 ± 0.09 |
| E2: T0, class names masked | 78.24 | 27.79 | 28.13 |
| E2: UB, class names masked | 67.05 ± 0.41 | 28.42 ± 0.09 | 28.51 ± 0.05 |
| E2: F0, class names masked | 86.27 ± 0.48 | 74.02 ± 0.55 | 73.61 ± 0.59 |
| E2: F1, class names masked | 89.01 ± 0.21 | 74.91 ± 0.12 | 74.38 ± 0.12 |
| E2: F2, class names masked | 90.50 ± 0.12 | 75.39 ± 0.11 | 74.88 ± 0.15 |

## Support / conflict (preliminary, seed 42)

Groups by unimodal correctness: support = UA and UB right; conflict_I = only the image model right; conflict_T = only the text model right.

| group | items | share | f0_late_fusion acc. | f1_concat acc. | f2_crossmodal acc. | f0_late_fusion follows image | f1_concat follows image | f2_crossmodal follows image |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| support | 13,785 | 61.0 % | 100.0 | 99.8 | 99.8 | 100.0 | 99.8 | 99.8 |
| conflict_I | 2,219 | 9.8 % | 92.7 | 72.1 | 77.6 | 92.7 | 72.1 | 77.6 |
| conflict_T | 5,503 | 24.4 % | 96.8 | 97.6 | 97.5 | 3.2 | 1.3 | 1.3 |
| both_wrong_agree | 143 | 0.6 % | 0.0 | 14.0 | 14.0 | 100.0 | 77.6 | 79.0 |
| both_wrong_disagree | 944 | 4.2 % | 1.3 | 26.3 | 31.1 | 78.0 | 15.6 | 17.2 |

The same table split by class-name mention and by title seen / unseen is in support_conflict.csv.
