# Assignment 3 - Final Report (M3): UPMC Food-101 Image-Title Classification

Deep Learning and Its Applications (CO3133), Semester 261 - Instructor: Lê Thành Sách
Group G-roundhog - Phan Bá Thanh (2353084), Trần Công Hoàng Phước (2352966)
Task option: image-text classification (spec 6.2). Proposal (M1): [PROPOSAL.md](../PROPOSAL.md), approved. M2 draft: [REPORT_M2.md](REPORT_M2.md), approved.

The full report is the PDF [a3-report/main.pdf](a3-report/main.pdf), with its LaTeX source in `docs/a3-report/`. This Markdown version holds the same results. For the data, the pair audit, the EDA and the split (Part 1), it points to the M2 report, which already covers them in full.

**Status (spec 6.13, M3).** The improved fusion and the ablations are complete. Every trained model has three seeds (42, 123, 2026).

| ID | Model |
| --- | --- |
| UA | image only, ResNet-50 |
| UB | text only, DistilBERT |
| F0 | late fusion |
| F1 | concat fusion |
| **F2** | improved cross-modal fusion |
| K0 | keyword rule (text reference) |
| T0 | TF-IDF + logistic regression (text reference) |

The ablations of F2 are:
- **E1:** auxiliary losses on or off.
- **E2:** class names masked in every title.
- **E4:** modality dropout on or off.
- **E-mod:** modality-removal curves.

The analyses are:
- support/conflict over three seeds;
- Grad-CAM and `[FUS]` attention;
- temperature scaling;
- a manual error taxonomy of 100 conflict items.

**Traceability.**
- Split: [data/splits/split.json](../data/splits/split.json).
- Code commits:
  - UA, UB, F1: `cfc509a`.
  - F2, the ablations and T0: `b51d3fb`.
  - Analyses: `850b6bb` and later.
- Every run's `env.json` records its commit, always with a clean working tree.
- The checkpoints are listed in section 6.

---

## 1. Problem and data (summary; details in [REPORT_M2.md section 1](REPORT_M2.md#1-problem-and-data))

**Task.** Single-label, 101-way classification of recipe web pages. The input is one image found on the page plus the page **title**. The data is UPMC Food-101, Kaggle mirror `gianmarco96/upmcfood101`; this mirror has titles only, no full page text.

**Pairs.**

| Split | Complete pairs (image + valid title) |
| --- | ---: |
| train | 62,771 |
| val | 4,824 |
| test | 22,594 |

- The split is grouped by image-duplicate clusters (md5 + pHash).
- The natural missing-text test set has 121 items.
- A manual audit of 200 pairs found 81.5 % valid pairs and 0.5 % wrong labels.

**Two leaks, both reported, not hidden.**
1. **Class names in titles.** 79 % of the titles name their class, because they come from "\<class\> recipe" searches.
2. **Repeated items across the official lists.**
   - 39.4 % of the test titles also occur in train/val.
   - 14.5 % of the test images have a train duplicate.

Every model is therefore also evaluated on a **dedup** view and a **title-unseen** view. E2 removes the class names entirely.

## 2. Methodology

### 2.1 Models

| ID | Model | Params |
| --- | --- | ---: |
| K0 | keyword rule (lexicon of class names, synonyms) | 0 |
| T0 | TF-IDF (word 1-2-grams + char 3-5-grams, min_df 2) + multinomial logistic regression, C on val | 13.6 M |
| UA | ResNet-50 (ImageNet V2) pooled 2048 -> linear 101 | 23.71 M |
| UB | DistilBERT-base-uncased `[CLS]` -> dropout -> linear 101 | 66.44 M |
| F0 | late fusion: p = w p_UA + (1 - w) p_UB, w on val (0.45 / 0.45 / 0.50) | 0 extra |
| F1 | concat fusion: 2048 -> 512 + LN, 768 -> 512 + LN, concat -> MLP -> 101, end to end | 91.89 M |
| **F2** | cross-modal transformer fusion + auxiliary losses + modality dropout (below) | 97.98 M |

### 2.2 F2 - improved fusion ([src/models/classifiers.py](../src/models/classifiers.py), [configs/upmc/f2_crossmodal.yaml](../configs/upmc/f2_crossmodal.yaml))

The M2 analysis showed that F1 **under-uses the image**:
- the title alone fits the training set;
- without the title, F1 falls to 53 % accuracy, below the image-only model (71 %).

F2 attacks this with three components, all fixed in the proposal before any F2 run.

1. **Token-level cross-modal attention.**
   - **Image tokens:** the ResNet-50 `layer4` map gives 49 region tokens. Each is projected to 512 and gets a learned 2-D position and a type embedding.
   - **Title tokens:** the DistilBERT hidden states are projected to 512 and get a type embedding.
   - **Fusion token:** a learned `[FUS]` token is prepended.
   - **Fusion:** 2 pre-LN transformer layers (8 heads, FFN 2048, dropout 0.1) with padding masked. Then `[FUS]` -> LayerNorm -> 101.
2. **Auxiliary unimodal losses.** Linear heads classify the pooled image feature and the text `[CLS]`. The loss is L = CE(fused) + λ [CE(image head) + CE(text head)], with λ = 0.5.
3. **Modality dropout with learned `[MISSING]` tokens.**
   - In training, each complete pair loses one modality with probability p = 0.15 (image or title, 50/50, never both).
   - All tokens of the dropped modality are replaced by one learned `[MISSING_IMG]` / `[MISSING_TXT]` token; the other positions are masked.
   - At test time, a missing modality is handled the same way.

**Schedule.**
- Epoch 1: both encoders frozen, only the 8.1 M new parameters learn.
- Epochs 2-10: full fine-tuning with the F1 learning rates: image 1e-4, text 3e-5, new layers 5e-4.
- Warm-up: 15 % of the steps.
- Checkpoint: best val macro-F1.

### 2.3 Controlled experiments

| ID | Change (everything else identical, 3 seeds) | Question |
| --- | --- | --- |
| E1 | λ = 0. The aux heads are trained on detached features, so they become linear probes that measure each encoder without changing it | Do the auxiliary losses keep the image branch alive? |
| E2 | every class-name mention replaced by `[MASK]` in train, val and test titles; UB, T0, F0, F1, F2 retrained | How much depends on the label leaking into the title? Which fusion uses the image best when it does not? |
| E4 | p = 0 (no modality dropout) | Does modality dropout help with missing modalities? Does it cost anything on complete pairs? |
| E-mod | at test time, remove the title (or image) from 0/25/50/75/100 % of the complete pairs | How fast does each model degrade? |

- **E2 masking:** it changes 79.8 % of the training titles. After masking, the lexicon finds no class name in any title.
- **E-mod fallbacks for a removed modality:**
  - F1 zero-fills it.
  - F2 uses its `[MISSING]` token.
  - F0 falls back to the other unimodal model.
  - UA and UB predict the most frequent training class.

### 2.4 Protocol (unchanged from M2)

- **Metrics:** macro-F1 (primary), top-1, top-5, ECE (15 bins).
- **Decision rule** (registered in the proposal): A beats B only if:
  1. the mean macro-F1 difference is larger than the larger seed std, **and**
  2. McNemar on the seed-42 predictions gives p < 0.05.
- **Hardware:** lab RTX A6000 (48 GB, shared), Ubuntu 24.04, PyTorch 2.12 + CUDA 13.0, bf16 autocast, batch 64, 16 workers.

## 3. Results

### 3.1 Main comparison (test complete pairs, mean ± std over 3 seeds)

| Model | Accuracy | Macro-F1 | Top-5 | ECE | Macro-F1 dedup | Macro-F1 title unseen |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| K0 keyword rule | 76.95 | 83.56 | 77.41 | 2.97 | 83.81 | 82.89 |
| T0 TF-IDF + LR | 87.12 | 87.26 | 91.23 | 2.71 | 87.36 | 82.57 |
| UA image | 70.73 ± 0.09 | 70.80 ± 0.06 | 86.51 ± 0.16 | 10.75 ± 0.19 | 67.72 ± 0.03 | 70.59 ± 0.12 |
| UB text | 86.42 ± 0.91 | 86.80 ± 0.25 | 90.71 ± 0.69 | 2.84 ± 1.46 | 86.96 ± 0.07 | 82.57 ± 0.44 |
| F0 late fusion | **93.93 ± 0.23** | **93.86 ± 0.24** | **97.71 ± 0.07** | 15.98 ± 0.78 | **93.71 ± 0.27** | **92.03 ± 0.32** |
| F1 concat fusion | 92.85 ± 0.06 | 92.80 ± 0.06 | 97.15 ± 0.04 | 4.69 ± 0.02 | 92.80 ± 0.07 | 90.28 ± 0.03 |
| **F2 cross-modal fusion** | 93.58 ± 0.07 | 93.53 ± 0.07 | 97.69 ± 0.04 | **2.88 ± 0.09** | 93.49 ± 0.06 | 91.05 ± 0.14 |

| A vs B | Δ macro-F1 (pp) | larger std | McNemar b / c, p | verdict |
| --- | ---: | ---: | --- | --- |
| UB vs T0 | -0.46 | 0.25 | 260 / 655, 9e-39 | T0 better |
| F2 vs UB | +6.73 | 0.25 | 2036 / 168, <1e-300 | F2 better |
| F1 vs F0 | -1.06 | 0.24 | 426 / 609, 1.5e-8 | F0 better |
| **F2 vs F1** | **+0.73** | 0.07 | 439 / 278, 2.3e-9 | **F2 better** |
| **F2 vs F0** | -0.33 | 0.24 | 465 / 487, 0.50 | **no decision** |

The full table, with every pair, is in [outputs/comparison/summary.md](../outputs/comparison/summary.md).

- **Text dominates the unimodal models.**
  - UB is 16 pp above UA.
  - The TF-IDF model T0 is slightly better than DistilBERT on the full test set. On unseen titles the two are equal (82.6 macro-F1). T0 memorises repeated titles better.
- **Every fusion beats every unimodal model by 6-7 pp**, on all three test views.
- **F2 beats F1** (+0.73 pp; +0.77 pp on unseen titles).
- **F2 does not beat the late fusion F0.** The difference is within the seed spread and McNemar is flat. With two strong unimodal models and informative titles, averaging probabilities is a hard baseline.

### 3.2 E1 - auxiliary losses

| Model | Macro-F1 | Image head acc. | Text head acc. | Acc. title removed | Acc. image removed | Missing-text set acc. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| F2 (λ = 0.5) | **93.53 ± 0.07** | **69.7 ± 0.4** | **87.1 ± 0.0** | **65.3 ± 0.9** | 86.7 ± 0.2 | 61.4 ± 0.5 |
| E1 (λ = 0) | 93.07 ± 0.08 | 62.3 ± 0.1 | 76.8 ± 1.7 | 58.6 ± 0.8 | 86.3 ± 0.1 | 53.7 ± 3.0 |

- **Verdict: the auxiliary losses help.** Macro-F1 gains 0.46 pp (larger std 0.08, McNemar p = 3e-7).
- **They keep the encoders discriminative.**
  - F2's image head reaches 69.7 %, close to the standalone UA (70.7 %).
  - A linear probe on E1's image features reaches only 62.3 %.
  - This is the "modality laziness" found in F1 in M2, now measured directly.
- **The benefit is largest when the title is missing:** +6.7 pp with the title removed.

### 3.3 E4 - modality dropout

| Model | Macro-F1 | Acc. title removed | Acc. image removed | Missing-text set, as is | Missing-text set, [MISSING] |
| --- | ---: | ---: | ---: | ---: | ---: |
| F2 (p = 0.15) | 93.53 ± 0.07 | **65.3 ± 0.9** | 86.7 ± 0.2 | 61.4 ± 0.5 | 60.6 ± 2.4 |
| E4 (p = 0) | 93.45 ± 0.06 | 60.6 ± 0.4 | 86.5 ± 0.0 | 58.1 ± 1.9 | 57.9 ± 1.4 |

- **No decision on complete pairs** (+0.08 pp, p = 0.54). The modality dropout costs nothing there.
- **It helps when the title is missing:** +4.7 pp.
- **F2 still stays below the image model without the title** (65.3 vs 70.7 %). F0, which falls back to UA, remains the most robust to a missing title.

### 3.4 E2 - class names masked

| Model | Macro-F1 as is | Macro-F1 masked | Δ (pp) | Gain over UB, as is | Gain over UB, masked |
| --- | ---: | ---: | ---: | ---: | ---: |
| T0 TF-IDF | 87.26 | 47.79 | -39.5 | - | - |
| UB text | 86.80 ± 0.25 | 43.70 ± 0.16 | -43.1 | - | - |
| F0 late | **93.86 ± 0.24** | 78.85 ± 0.17 | -15.0 | 7.1 | 35.2 |
| F1 concat | 92.80 ± 0.06 | 80.43 ± 0.14 | -12.4 | 6.0 | 36.7 |
| F2 cross-modal | 93.53 ± 0.07 | **81.31 ± 0.02** | -12.2 | 6.7 | **37.6** |

- **Half of the text signal is the class name.**
  - **Masking halves the text models.** On titles that never named the class, UB scores the same with or without masking (39.2 vs 39.7 %). The masking removes the name and nothing else.
  - **The rest of a masked title is still worth 44 %:** ingredients, cooking words, sites and repeated titles. The masked text model is 67 % accurate on titles seen in train and 28 % on unseen ones.
- **On masked titles, F2 is the best fusion.**
  - +0.88 pp over F1 (p = 2e-4) and +2.46 pp over F0 (p = 2e-28).
  - It has the smallest seed spread.
  - When no modality is decisive, token-level cross-attention combines partial evidence better than averaging two probability vectors. The F0 > F1 ranking of plain titles is reversed.

### 3.5 E-mod - modality-removal curves ([removal_curves.png](../outputs/analysis/removal_curves.png))

Macro-F1 with the title removed from a fraction of the test pairs:

| Model | 0 % | 25 % | 50 % | 75 % | 100 % |
| --- | ---: | ---: | ---: | ---: | ---: |
| UA image | 70.8 | 70.8 | 70.8 | 70.8 | 70.8 |
| UB text | 86.8 | 73.4 | 57.3 | 34.1 | 0.0 |
| F0 late | 93.9 | 88.1 | 82.2 | 76.7 | **70.8** |
| F1 concat | 92.8 | 82.9 | 73.1 | 63.0 | 52.5 |
| F2 cross-modal | 93.5 | 86.6 | 79.5 | 72.5 | 65.3 |
| E1 (no aux.) | 93.1 | 84.5 | 76.0 | 67.3 | 58.5 |
| E4 (no mod. dropout) | 93.4 | 85.3 | 77.1 | 68.8 | 60.1 |

- **The curves are linear by construction** (random removal), so the end points carry the information.
- **Removing the image costs every fusion about 7 pp.** All fusions end at UB's level (86.3-86.8).
- **Removing the title separates the models.** F2's two mechanisms close more than half of F1's gap to the image model.
- **Natural missing-text set (121 items), accuracy:**
  - UA 65.6, F0 68.9, F2 61.4, F1 51.8.
  - The F1 value comes from M2's evaluation, which zero-filled the text in both variants.

### 3.6 Support / conflict over 3 seeds ([support_conflict_seeds.png](../outputs/analysis/support_conflict_seeds.png))

| Group (UA, UB of the same seed) | Share | F0 | F1 | F2 | E1 | E4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| support (both right) | 61.7 % | 100.0 | 99.8 | 99.8 | 99.8 | 99.8 |
| conflict-I (only image right) | 9.0 % | **90.6 ± 4.5** | 69.0 ± 2.7 | 75.3 ± 2.0 | 70.9 ± 1.3 | 74.7 ± 1.6 |
| conflict-T (only text right) | 24.7 % | 97.1 ± 0.9 | 97.7 ± 0.1 | 97.7 ± 0.2 | 97.9 ± 0.2 | 97.8 ± 0.2 |
| both wrong, same prediction | 0.7 % | 0.0 | 10.7 | 10.6 | 10.7 | 10.0 |
| both wrong, different predictions | 3.9 % | 0.7 | 21.3 | **24.0** | 22.6 | 23.2 |

- **The M2 conclusions hold over three seeds.** The fusion gain comes from the conflict groups, and a misleading title (conflict-I) is the weak spot of the trained fusions.
- **F2 recovers 6.3 pp more of conflict-I than F1.** Most of that comes from the auxiliary losses (E1). F0 still recovers the most.
- **Only the trained fusions solve some "both wrong" items.** A late fusion can only pick one of the two unimodal answers.
- **With masked titles, the balance shifts.** Conflict-I grows to 40 % of the items. F2-mask is best on conflict-T (85.7 vs 75.9 % for F0-mask).

### 3.7 Grad-CAM and `[FUS]` attention ([outputs/analysis/qualitative_*.png](../outputs/analysis/))

Mean share of `[FUS]` attention on the image tokens (last fusion layer, seed 42):

| Model | all | support | conflict-I | conflict-T | both wrong |
| --- | ---: | ---: | ---: | ---: | ---: |
| F2 | 0.805 | 0.792 | **0.909** | 0.778 | 0.891 |
| E1 (no aux.) | 0.756 | 0.745 | 0.873 | 0.720 | 0.847 |
| E4 (no mod. dropout) | 0.883 | 0.883 | 0.930 | 0.858 | 0.917 |
| F2-mask | 0.922 | 0.928 | 0.927 | 0.907 | 0.912 |

- **F2 moves its attention to the image when the title is uninformative.**
  - Titles such as "Account Suspended", "Hello" or "In Rome" get 0-4 % of the attention, and F2 classifies from the image.
  - Titles that name the dish put the most attention on that word ("quesadilla", "macaroni").
- **Compare the shares between groups.** The image has 49 tokens against about 15 for a title, so uniform attention would already give about 0.75 to the image.
- **Grad-CAM** usually highlights the food. Its failures are mixed platters and text overlaid on the image.

### 3.8 Error taxonomy of 100 conflict items ([data/audit/conflict_review.csv](../data/audit/conflict_review.csv))

**Protocol.**
- 50 conflict-I and 50 conflict-T items, drawn at random from seed 42.
- Each item was labelled with the main cause why the failing unimodal model is wrong: Phan Bá Thanh labelled items 1-50 and Trần Công Hoàng Phước items 51-100.
- Scores are produced by [scripts/conflict_review.py](../scripts/conflict_review.py) `score` and saved in [conflict_review_summary.json](../data/audit/conflict_review_summary.json).

| Group (failing model) | Cause | Items | Share (Wilson 95 % CI) | F0 right | F1 right | F2 right |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| conflict-I (text wrong) | title generic (no dish information) | 42 | 84 % (71-92) | 100 % | 74 % | 93 % |
| | title names another dish | 8 | 16 % (8-29) | 50 % | 38 % | 50 % |
| conflict-T (image wrong) | visually similar to another dish | 34 | 68 % (54-79) | 94 % | 94 % | **100 %** |
| | image unrelated | 13 | 26 % (16-40) | 100 % | 100 % | 100 % |
| | wrong label | 2 | 4 % (1-13) | 100 % | 100 % | 100 % |
| | other | 1 | 2 % (0-10) | 100 % | 100 % | 100 % |

- **The text model fails mostly on generic titles**, not on wrong ones. F0 and F2 fix almost all of these.
- **The hard case is a title that names another dish.** For example, "French Bread Pizza" on a `pizza` image is read as `garlic_bread`. Every fusion gets only about half of these right.
- **The image model fails mostly on look-alike dishes and unusual plating.** F2 is the only model that solves every such reviewed item.
- **Wrong labels are rare (2 %).**
- **28 % of the reviewed titles also occur in train/val.**
  - Some text successes are memorised: "- quoteko.com" occurs 80 times in train/val, 77 of them as `breakfast_burrito`, so UB predicts that class with 98 % confidence for a photo of baked oatmeal.
  - Others are lucky guesses with a confidence of a few percent.
- **Limitations:** a small sample (8 items for "another dish") and one rater per item.

### 3.9 Calibration (temperature scaling, T fitted on val)

| Model | T | ECE before | ECE after |
| --- | ---: | ---: | ---: |
| UA | 1.62 | 10.75 | 4.22 |
| UB | 1.30 | 2.84 | 1.01 |
| F0 | 0.61 | 15.98 | 6.46 |
| F1 | 1.91 | 4.69 | 0.90 |
| **F2** | 1.40 | **2.88** | **0.45** |

- **F2 is the best-calibrated model.**
- **F0 stays the worst.** Its averaged probabilities are under-confident, and one temperature cannot fix that.

### 3.10 Compute cost (RTX A6000, shared; bf16)

| Model | Params | GMACs/item | Train time/run | Train items/s | Inference items/s (batch 128) | Latency (batch 1) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| T0 | 13.6 M | - | 4.4 min (CPU) | - | - | - |
| UA | 23.71 M | 4.09 | 14.4 min | 728 | 2,793 | 4.7 ms |
| UB | 66.44 M | 0.64 | 2.0 min | 2,420 | 17,105 | 2.7 ms |
| F1 | 91.89 M | 4.73 | 18.4 min | 568 | 2,385 | 7.7 ms |
| F2 | 97.98 M | 5.06 | 18.6 min | 526 | 2,220 | 8.5 ms |

- **F2 costs 7 % more compute than F1.**
- **Total GPU time:**
  - M2: 9 runs in 2.0 GPU-hours.
  - M3: 18 runs (4.75 h of training) plus evaluation, removal predictions and Grad-CAM, in about 5.7 GPU-hours.
  - Whole assignment: about 7.7 GPU-hours.

## 4. Deviations from the proposal

| Proposal | Implementation | Reason |
| --- | --- | --- |
| Kaggle GPU for all runs | lab RTX A6000 (shared) for training; Kaggle CPU for data preparation | Kaggle GPU quota ran out |
| E-mod with retraining at several drop rates | test-time removal curves for every model; the effect of training with missing modalities is measured by E4 | one controlled factor per experiment |
| optional E3 (frozen encoders) | not run | time; E4 and T0 chosen instead |
| - | title-unseen test view | title leakage found in M2 |

The M2 deviations (validation size, pHash duplicates, K0 lexicon, timing runs, five audit verdicts) are listed in [REPORT_M2.md](REPORT_M2.md).

## 5. Conclusions

1. **Multimodality pays off.** Every fusion is 6-7 pp above the best single modality. The gain comes from the 34 % of items where exactly one modality is right, and it holds on the leakage-controlled views.
2. **The title is strong but leaky.**
   - 79 % of the titles name the class; masking these names halves the text model.
   - 39 % of the test titles repeat a training title.

   Claims about the text modality must therefore be checked on masked and unseen titles.
3. **F2 does what it was designed to do.**
   - It beats F1 on every view.
   - Its auxiliary losses keep the image encoder as strong as a standalone image model.
   - Its modality dropout adds robustness to a missing title at no cost.
   - Its attention moves to the image when the title is uninformative.
4. **F2 matches, but does not beat, the late fusion F0 on complete pairs.** F2 becomes the best model when the title is less direct (E2: +2.5 pp over F0), and it is the best-calibrated model. Which model to deploy:
   - **F0** when titles are informative and often missing;
   - **F2** when calibrated confidence matters or the text is weak.
5. **Remaining errors** are titles that name another dish, and look-alike dishes. Next steps:
   - a food-pretrained image encoder (A2);
   - image-text contrastive pretraining;
   - a split with titles deduplicated across train and test.

## 6. Links

- **Repository:** <https://github.com/pbt245/CO3133-A3>
- **Checkpoints** (selected checkpoint of every run, with SHA-256 checksums):
  - M2: <https://www.kaggle.com/datasets/bathanhphan/co3133-a3-m2-checkpoints>
  - M3: <https://www.kaggle.com/datasets/bathanhphan/co3133-a3-m3-checkpoints>
- **Assignment page:** <https://pbt245.github.io/CO3133-DLIA/assignments/assignment3.html>
- **Slides:** [docs/a3-slides/slides.pdf](a3-slides/slides.pdf)
- **Video:** `[video URL]`
- **AI usage disclosure:** see the PDF report.
