# Assignment 3 - M2 Draft Report: UPMC Food-101 Image-Title Classification

Deep Learning and Its Applications (CO3133), Semester 261 - Instructor: Lê Thành Sách
Group G-roundhog - Phan Bá Thanh (2353084), Trần Công Hoàng Phước (2352966)
Task option: image-text classification (spec 6.2). Proposal (M1): [PROPOSAL.md](../PROPOSAL.md), **approved**.

**Status of this draft (spec 6.13, M2).** Both unimodal baselines (UA image, UB text) and the simple fusion (F1 concat) are implemented, trained with **3 seeds** each and evaluated on the official test set. Two zero-training references are reported with them: the keyword rule K0 and the late fusion F0. The data pipeline, the pair-validity checks with the 200-pair manual audit, the grouped split, the EDA and a first support/conflict analysis are also done. The improved fusion F2, the ablations (E1, E2, E-mod with retraining), the error taxonomy, slides and video belong to M3.

All numbers below come from the committed split (`data/splits/split.json`, fingerprints train `32547835c88028e6`, val `5028de5edbd16bda`, test `1f8fa282b49dc603`) and from code commit `cfc509a` (recorded in every run's `env.json`, working tree clean).

---

## 1. Problem and data

### 1.1 Problem

Single-label, 101-way classification of **recipe web pages**. Each item is one image found on a web page plus the **title** of that page; the label is one of the 101 Food-101 dish classes. The two modalities are linked by the page they come from (same file id `<label>_<num>`), never paired at random (spec 6.4).

### 1.2 Dataset and pairs

Source: UPMC Food-101 (Wang et al., ICME-W 2015), Kaggle mirror `gianmarco96/upmcfood101`. The mirror holds images and **page titles only** (no full page text), so our text modality is the title.

Pairs are built and checked by [scripts/prepare_data.py](../scripts/prepare_data.py) (Kaggle CPU notebook `co3133-a3-prepare`, 26 min, log in [outputs/lab_logs/prepare_log_kaggle.txt](../outputs/lab_logs/prepare_log_kaggle.txt)):

| Criterion (proposal 2.3) | Rule | Result |
| --- | --- | --- |
| 1. Image valid | decodes, shorter side >= 64 px, pixel std > 2 | 90,701 of 90,704 (3 near-constant images = image-missing) |
| 2. Text valid | title exists and has >= 1 content word (>= 3 letters, not in a fixed generic-word list) | 90,192 of 90,704; 16 images have no title row, 496 titles fail the rule |
| 3. Same item | image and title share the id, label and official split | 90,688 title rows, all matched; 0 label or split mismatches |
| 4. Manual audit | 200 train pairs, 20 classes x 10 | 81.5 % valid; about 10 % uninformative titles, 9.5 % uninformative images, 0.5 % wrong labels (section 1.4) |

| Split | Items | Complete pairs | Text-missing | Image-missing | Test items with a train duplicate | In a cross-label cluster |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 63,128 | **62,771** | 355 (0.6 %) | 2 | - | 869 |
| val | 4,860 | **4,824** | 36 (0.7 %) | 0 | - | 75 |
| test | 22,716 | **22,594** | 121 (0.5 %) | 1 | 3,290 (14.5 %) | 296 |

All models are trained, selected and tested on **complete pairs**. The 121 text-missing test items form the natural missing-text set (section 3.4). Spec 6.5 threshold (>= 5,000 pairs, >= 5 classes) is met by a factor of 12.

### 1.3 EDA (spec 6.10; full tables in [outputs/eda/EDA.md](../outputs/eda/EDA.md))

**Text.** Titles are short: median 7 words / 13 DistilBERT tokens (p99 39 tokens); only 126 titles (0.14 %) exceed the 64-token limit. Train vocabulary: 29,847 lower-cased words. Titles are noisy web strings: HTML entities (decoded), site suffixes, some literally escaped UTF-8 bytes (`Soupe \xc3\xa0 l'oig`), JSON-like brackets. The most frequent invalid titles are `All Yummy Recipes`, `Recipes`, `Recipes on Pinterest`, `Easy`, `Food`. Figures: [text_lengths.png](../outputs/eda/text_lengths.png), [samples_pairs.png](../outputs/eda/samples_pairs.png).

**Label information in the title** (complete pairs, class-name lexicon of [src/data/text.py](../src/data/text.py): names, singular/plural forms, 82 hand-written synonyms for 50 classes):

| Split | Own class, exact name | Own class, lexicon | Another class only | No class mentioned | K0 keyword rule acc. |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 72.3 % | 78.8 % | 1.0 % | 20.2 % | 76.9 % |
| test | 72.4 % | 78.8 % | 0.9 % | 20.2 % | 76.8 % |

The search query ("<class> recipe") leaks the label into ~79 % of titles. Per class this ranges from 7 % (`hot_dog`) and 30 % (`beef_tartare`) to 97.5 % (`pancakes`) ([class_name_mention_per_class.png](../outputs/eda/class_name_mention_per_class.png)).

**Site names.** 64 % of titles end with a site segment (`| Taste of Home`, `on Pinterest`); 9,294 distinct sites in train. For the 96 sites with >= 50 train items, the majority class covers only 16 % of a site's items on average, so the site name alone is a weak class cue (exception: `popsugar food`, 31 %).

**Images.** Web photos with varied geometry: shorter side median 387 px (120-900), aspect ratio median 1.33 (0.09-8.73: banners and collages). Train channel mean [0.628, 0.529, 0.422] is brighter than ImageNet's [0.485, 0.456, 0.406] (white backgrounds of recipe sites); pretrained models still use ImageNet statistics. Classes are mildly imbalanced (max/min 1.21). Figures: [image_stats.png](../outputs/eda/image_stats.png), [samples_per_class.png](../outputs/eda/samples_per_class.png), [class_distribution.png](../outputs/eda/class_distribution.png).

**Duplicates.** 3,597 md5 links and 12,663 pHash pairs (Hamming <= 4) form 7,201 clusters with >= 2 items (16,498 items, largest 16). 524 clusters carry two or more labels (the same picture under `filet_mignon` and `steak`, [duplicates_cross_label.png](../outputs/eda/duplicates_cross_label.png)). 10,859 normalised titles repeat (4,873 across the official train/test lists).

### 1.4 Pair audit (criterion 4)

**Protocol.** We drew 10 random train pairs from each of 20 random classes (seed 42) with [scripts/audit_pairs.py](../scripts/audit_pairs.py). Each pair was judged by one member: Phan Bá Thanh judged pairs 1-100 and Trần Công Hoàng Phước pairs 101-200. The verdicts are in [data/audit/pair_audit.csv](../data/audit/pair_audit.csv) and the scores in [pair_audit_summary.json](../data/audit/pair_audit_summary.json).

| Verdict | Count | Rate | Wilson 95 % CI |
| --- | ---: | ---: | --- |
| valid | 163 | **81.5 %** | 75.5-86.3 % |
| image unrelated | 16 | 8.0 % | 5.0-12.6 % |
| text unrelated | 17 | 8.5 % | 5.4-13.2 % |
| both unrelated | 3 | 1.5 % | 0.5-4.3 % |
| wrong label | 1 | 0.5 % | 0.1-2.8 % |
| *image uninformative (image + both)* | 19 | 9.5 % | 6.2-14.4 % |
| *title uninformative (text + both)* | 20 | 10.0 % | 6.6-14.9 % |

**Findings.**
- **About one pair in five is noisy, and the noise sits in one modality at a time.** Only 1.5 % of the pairs are unrelated in both modalities. This is the raw material of the conflict groups (section 3.2): when one side is uninformative, the other side usually still carries the class.
- **The noise is concentrated in a few classes:**
  - `peking_duck`: 5 of its 10 images are unrelated.
  - `grilled_salmon`: 5 of its 10 titles are unrelated.
  - `hot_and_sour_soup`: 4 of its 10 titles are unrelated.
  - Seven of the 20 classes have no noisy pair.
- **The automatic text rule is lenient.** Criterion 2 flags only 2 of the 20 titles judged uninformative; it catches empty or generic titles ("Recipes") but not titles of other pages or other recipes.
  - The real share of uninformative titles is therefore about 10 %, not the 0.6 % "text-missing" rate of section 1.2.
  - Complete pairs still contain such titles, and the natural missing-text set understates real missingness.
  - 19 of the 20 uninformative titles do not mention the class, against 19 of the 180 informative ones. A missing class name is a strong but not sufficient signal of a useless title.
- **Wrong labels are rare (0.5 %).** This is much lower than the 6.7 % wrong labels we measured in the ETHZ Food-101 train set (A2). Here image and title come from the same query page, so they rarely agree on a different dish. The cross-label duplicate clusters of section 1.3 are the main source of label noise.
- **Limitations.** Each pair was judged by one rater, so there is no inter-rater agreement. The two halves cover different classes, so the members' rates (22 vs 15 noisy pairs) are not comparable. The sample covers 20 of the 101 classes.

### 1.5 Split and leakage prevention (spec 6.4)

- **Split unit = item** (one web page). Official train -> `StratifiedGroupKFold(n_splits=14, shuffle, seed 42)`, stratum = class, group = image-duplicate cluster, fold 0 = validation (4,860 items). No cluster is shared by train and val (asserted). The official test list is untouched.
- **Test leakage is reported, not hidden:** every test metric is also given on the **dedup** subset (19,270 complete pairs without a train-linked image and without cross-label clusters).
- **New in M2: title leakage.** 39.4 % of complete test pairs have a normalised title that also occurs in train/val: several images of one web page were split across the official lists, or the title is generic. The image clusters do not capture these links, and the official test list cannot be regrouped, so we report a third view: **title-unseen** test items (13,701). Section 3.3.

---

## 2. Methodology

### 2.1 Pipeline

Mirror -> `prepare_data.py` (validity, md5 + pHash clusters, split, images pre-resized to a 256 px shorter side, `upmc256.zip`) -> committed `data/splits`, `data/meta` -> `train.py` per model and seed -> `evaluate.py` (val + test predictions, subsets, missing-text set, modality removal, inference cost) -> `keyword_baseline.py` (K0), `late_fusion.py` (F0) -> `compare.py` (mean ± std, decision rule, support/conflict).

### 2.2 Models (spec 6.6, 6.7)

| ID | Model | Encoders and fusion | Params |
| --- | --- | --- | ---: |
| K0 | Keyword rule | most-mentioned class name in the title (lexicon); no mention -> most frequent train class | 0 |
| **UA** | Image only | ResNet-50 (ImageNet `IMAGENET1K_V2`) pooled 2048-d -> linear 101 | 23.71 M |
| **UB** | Text only | DistilBERT-base-uncased `[CLS]` 768-d -> dropout 0.1 -> linear 101 | 66.44 M |
| F0 | Late fusion | p = w p_UA + (1 - w) p_UB, w on a 0.05 grid chosen on val macro-F1 (w = 0.45 / 0.45 / 0.50) | 0 extra |
| **F1** | Concat fusion (intermediate, simple) | image 2048 -> 512 + LayerNorm, text 768 -> 512 + LayerNorm, concat 1024 -> 512 -> GELU -> dropout 0.1 -> 101, end-to-end | 91.89 M |

- **Alignment:** item-level, through the shared page id; titles are tokenised once per run (WordPiece, max 64, each batch trimmed to its longest title).
- **Losses:** one cross-entropy on the fused logits (F1) or the unimodal logits (UA, UB). Per-modality auxiliary losses are F2's contribution (M3, ablation E1).
- **Missing modality:** F1 multiplies the missing modality's projected vector by 0 (zero-fill); F0 falls back to UA.
- **Pretraining / fine-tuning:** both encoders start from public pretrained weights and are fully fine-tuned with discriminative learning rates.

### 2.3 Training setup (identical across models unless noted)

| Setting | Value |
| --- | --- |
| Image transforms | train `RandomResizedCrop(224, 0.35-1)` + flip + colour jitter 0.1; eval `Resize(256)` + `CenterCrop(224)` |
| Optimiser | AdamW; LR image encoder 1e-4, text encoder 3e-5, new layers 1e-3 (UA, UB) / 5e-4 (F1); weight decay 0.05 image + head, 0.01 text; no decay on biases and norms |
| Schedule | linear warm-up over 5 % of steps, cosine decay to 1 % |
| Epochs / batch | UA, F1: 10; UB: 5; batch 64; bf16 autocast; gradient clipping 1.0 |
| Selection | best val macro-F1 (complete pairs); early stopping after 3 epochs without improvement |
| Seeds | split 42; training 42, 123, 2026 |

### 2.4 Hardware and cost

Kaggle GPU quota ran out, so every training run used a **lab workstation: NVIDIA RTX A6000 (48 GB, shared with other users' inference servers), 64 CPU cores, Ubuntu 24.04, Python 3.13.13, torch 2.12.0+cu130, torchvision 0.27.0, transformers 5.23.0, CUDA 13.0**, 16 data-loading workers. The runner ([scripts/remote/](../scripts/remote/)) runs in tmux, picks the GPU with the most free memory (GPU 1, 32 GB free) and resumes after a crash (no retry was needed). Log: [outputs/lab_logs/](../outputs/lab_logs/).

| Model | Train time / run | Epochs run (best) | Train throughput | Inference (batch 128) | Latency (batch 1) | GMACs / item |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| UA | 14.4 min | 10 (9) | 728 img/s | 2,793 items/s | 4.7 ms | 4.09 |
| UB | 2.0 min | 4-5 (1-5) | 2,420 items/s | 17,105 items/s | 2.7 ms | 0.64 |
| F1 | 18.4 min | 10 (9) | 568 items/s | 2,385 items/s | 7.7 ms | 4.73 |

The 9 runs plus evaluation took **2.0 GPU-hours** (11:43-13:43). Throughput is about 2.5x lower than the proposal assumed for the Kaggle RTX PRO 6000 (shared, older GPU), so the M3 plan in the proposal (about 3.3 GPU-h on that GPU) should be read as about 6-8 GPU-h on this machine.

---

## 3. Results (test, complete pairs, mean ± std over 3 seeds)

Full tables: [outputs/comparison/summary.md](../outputs/comparison/summary.md); per run: `outputs/runs/upmc/<model>/seed<s>/`.

### 3.1 Main comparison

| Model | Accuracy | Macro-F1 | Top-5 | ECE | Macro-F1 dedup | Macro-F1 title-unseen |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| K0 keyword rule | 76.95 | 83.56 | 77.41 | 2.97 | 83.81 | 82.89 |
| UA image | 70.73 ± 0.09 | 70.80 ± 0.06 | 86.51 ± 0.16 | 10.75 ± 0.19 | 67.72 ± 0.03 | 70.59 ± 0.12 |
| UB text | 86.42 ± 0.91 | 86.80 ± 0.25 | 90.71 ± 0.69 | 2.84 ± 1.46 | 86.96 ± 0.07 | 82.57 ± 0.44 |
| F0 late fusion | **93.93 ± 0.23** | **93.86 ± 0.24** | **97.71 ± 0.07** | 15.98 ± 0.78 | **93.71 ± 0.27** | **92.03 ± 0.32** |
| F1 concat fusion | 92.85 ± 0.06 | 92.80 ± 0.06 | 97.15 ± 0.04 | 4.69 ± 0.02 | 92.80 ± 0.07 | 90.28 ± 0.03 |

K0's macro-F1 is above its accuracy because all 20 % of titles without a class name fall into one fallback class: one class gets a very low precision, all others a high one.

**Decision rule** (proposal 4: mean macro-F1 difference > larger seed std **and** McNemar p < 0.05 on seed 42):

| A vs B | Δ macro-F1 (pp) | Larger std | McNemar (A only right / B only right) | Verdict |
| --- | ---: | ---: | --- | --- |
| UB vs K0 | +3.24 | 0.25 | 2,031 / 128, p < 1e-300 | UB better |
| UB vs UA | +16.00 | 0.25 | 5,503 / 2,219, p = 2e-305 | UB better |
| F0 vs UB | +7.06 | 0.25 | 2,068 / 178 | F0 better |
| F1 vs UB | +6.00 | 0.25 | 1,868 / 161 | F1 better |
| F1 vs UA | +22.00 | 0.06 | 5,641 / 650 | F1 better |
| F1 vs F0 | -1.06 | 0.24 | 426 / 609, p = 1.5e-8 | **F0 better** |

Findings:
1. **Text dominates, as expected** (hypothesis E0): UB beats UA by 16 pp macro-F1. The gap is smaller than the 24 pp reported for full page text by Kiela et al. (2019), consistent with titles carrying less information than full text.
2. **UB is more than a keyword matcher**: +9.5 pp accuracy over K0, almost all of it on titles without a class name (39.2 % vs 0.5 % correct).
3. **Both fusions clearly beat the best unimodal model** (+6 to +7 pp macro-F1), on the full, the dedup and the title-unseen test sets.
4. **The simple trained fusion F1 loses to the zero-cost late fusion F0** by 1.06 pp (significant). Section 3.2 explains why.

### 3.2 Support / conflict analysis (spec 6.11, preliminary, seed 42)

Test items grouped by the correctness of the unimodal models ([support_conflict.png](../outputs/comparison/support_conflict.png), [support_conflict.csv](../outputs/comparison/support_conflict.csv)):

| Group | Items | Share | F0 acc. | F1 acc. | F1 follows the image prediction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Support (UA and UB right) | 13,785 | 61.0 % | 100.0 | 99.8 | 99.8 % |
| Conflict-I (only image right) | 2,219 | 9.8 % | **92.7** | 72.1 | 72.1 % |
| Conflict-T (only text right) | 5,503 | 24.4 % | 96.8 | **97.6** | 1.3 % |
| Both wrong, same prediction | 143 | 0.6 % | 0.0 | 14.0 | 77.6 % |
| Both wrong, different predictions | 944 | 4.2 % | 1.3 | 26.3 | 15.6 % |

- **Support** cases are solved by every model. **The fusion gain comes almost entirely from the conflict groups**, as the proposal expected: F0 and F1 recover 93-98 % of the conflict items.
- **Conflict-I is where F1 fails.** When the title misleads, F1 still follows the text in 28 % of the items, while F0 recovers 92.7 %. Without the class name in the title, conflict-I is 44.9 % of items, and F1 solves only 72.2 % of them vs 93.9 % for F0.
- **Evidence for modality laziness in F1:**
  - F1 reaches 99.7 % train accuracy; the text side alone can fit the training titles.
  - With the text removed at test time, F1 drops to 53.3 % accuracy, far below UA alone (70.7 %); with the image removed it keeps 86.3 % (UB: 86.4 %).

  The image branch inside F1 is therefore weaker than a standalone image model. This is the failure the proposal's F2 design targets (auxiliary unimodal losses, E1; modality dropout, E4).
- On the other hand, F1 is the only model that rescues some "both wrong" items (14-26 %), because it can combine partial evidence from both modalities.

### 3.3 Leakage checks

| View | UA macro-F1 | UB macro-F1 | F1 macro-F1 |
| --- | ---: | ---: | ---: |
| Full test (22,594) | 70.80 | 86.80 | 92.80 |
| Dedup (image), 19,270 | 67.72 (-3.1) | 86.96 (+0.2) | 92.80 (0.0) |
| Title unseen in train, 13,701 | 70.59 (-0.2) | 82.57 (-4.2) | 90.28 (-2.5) |

- **Image duplicates inflate the image model** by 3.1 pp. They do not affect the text model.
- **Repeated titles inflate the text model.** UB accuracy is 91.7 % on titles seen in train vs 83.0 % on unseen titles. The image model is a control here: 70.2 % vs 71.1 %.
- The ranking and every decision above hold on both leakage-controlled views. For M3 we will report the title-unseen subset next to the main metric.

### 3.4 Missing modality (natural missing-text set, 121 test items)

| Model | Accuracy |
| --- | ---: |
| UA (image, unaffected) | 65.56 ± 1.91 |
| UB (uninformative title) | 6.34 ± 1.72 |
| F0 with the title as is | 68.87 ± 2.66 |
| F0 falling back to UA | 65.56 ± 1.91 |
| F1 with the text zero-filled | 51.79 ± 1.72 |

- F1 was never trained with a missing modality, so zero-filling is out of distribution and it loses 14 pp against UA.
- The M2 evaluation code zero-filled F1 in both of its "missing-text" columns (these items carry `has_txt = 0`). The "as is" variant (title fed anyway) is fixed in the code for M3.
- On the complete test pairs, removing the text entirely costs F1 39.5 pp (92.9 -> 53.3).

### 3.5 Calibration

UB (ECE 2.8) and F1 (4.7) are well calibrated. UA is over-confident (10.8; 1,091 errors with confidence > 0.9 on seed 42). F0 is **under-confident** (16.0): averaging two softmax outputs that disagree spreads the probability mass. Temperature scaling on val is a cheap M3 addition.

### 3.6 Early error analysis (taxonomy in M3)

- **UA** confuses visually similar dishes, as in A2: `ice_cream` ↔ `frozen_yogurt` (45 + 33 test items), `filet_mignon` ↔ `steak` (32 + 24), `gyoza` → `dumplings`. `hot_dog` is its worst class (F1 0.27): web "hot dog recipe" images are often toppings, buns or party spreads.
- **UB** uses `beef_tartare` as a "don't know" class: its five largest confusions all predict `beef_tartare` (precision 0.22, recall 0.70). Its training titles rarely name the dish (30 %) and look generic or foreign-language, so uninformative test titles drift to it.
- **Image better than text:** `cheese_plate`, `beef_tartare`, `beef_carpaccio`, `croque_madame`. **Text better than image:** `hot_dog` (+57 pp), `ice_cream`, `frozen_yogurt`, `steak`, `pork_chop`, `ravioli`.
- **F1**'s remaining confusions are genuinely ambiguous pairs: `tuna_tartare` ↔ `beef_tartare`, `sushi` ↔ `sashimi`.
- Qualitative grids: `outputs/runs/upmc/<model>/seed42/eval/examples_*.png`.

### 3.7 Limitations of this draft

- One test protocol is fixed by the dataset: 14.5 % of test items have a train duplicate and 39 % share a title with train. We report leakage-controlled views but cannot regroup the official lists.
- Validation is grouped only by image clusters, so it shares titles with train like the test set does. Model selection is affected equally for all models.
- The support/conflict analysis uses seed 42 only; M3 repeats it over seeds and adds attention / Grad-CAM evidence.
- Training ran on a shared GPU; throughput numbers depend on the other jobs at that time (GPU usage log: [outputs/lab_logs/gpu_usage.csv](../outputs/lab_logs/gpu_usage.csv)).

---

## 4. Deviations from the proposal

| Proposal | M2 | Reason |
| --- | --- | --- |
| Kaggle RTX PRO 6000 | Lab RTX A6000 (shared) for training; Kaggle CPU for data preparation | Kaggle GPU quota exhausted |
| Val ≈ 5,000 items | 4,860 (n_splits = round(67,988 / 5,000) = 14, as specified) | formula in proposal 3.1 |
| Test duplicates ≥ 1,253 (md5) | 3,290 with pHash | near-duplicates added as planned |
| K0 on exact class names (71.2 %) | K0 with the lexicon (76.8 %) | the plurals and synonyms announced in proposal 2.4 |
| Titles pre-tokenised into a `.npy` cache | tokenised at dataset construction (seconds) | the cache was unnecessary |
| 1-epoch timing runs | 200-step timing runs before the main runs | enough to measure throughput and catch errors |
| Audit verdicts: 4 categories | 5 (adds "both unrelated") | image and text can both be off |
| - | new test view: title-unseen items | leakage found in M2 (section 3.3) |

## 5. Reproduction

See [README.md](../README.md) for full commands. In short:

```bash
# data (Kaggle CPU notebook co3133-a3-prepare, mirror attached)
python scripts/prepare_data.py --out /kaggle/working/prepare --resized /kaggle/working/upmc256.zip
python scripts/eda.py --split-dir .../data/splits --meta-dir .../data/meta --out .../eda
# training + evaluation (any CUDA machine; lab runner: bash scripts/remote/launch.sh)
python scripts/run_experiments.py --models ua_resnet50 ub_distilbert f1_concat --seeds 42 123 2026
python scripts/keyword_baseline.py && python scripts/late_fusion.py --seeds 42 123 2026 && python scripts/compare.py
```

Checkpoints (best.pt of all 9 runs, 2.1 GB, sha256 in `CHECKPOINTS_SHA256.txt`): private Kaggle dataset `bathanhphan/co3133-a3-m2-checkpoints` (shared on request; public release with M3).
