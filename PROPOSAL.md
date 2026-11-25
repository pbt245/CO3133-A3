# Assignment 3 - Dataset Proposal (Milestone M1)

**Course:** Deep Learning and Its Applications (CO3133), Semester 261 - Instructor: Lê Thành Sách
**Group:** G-roundhog - Phan Bá Thanh (2353084), Trần Công Hoàng Phước (2352966)
**Task option:** Image-text classification (multimodal)
**Submitted:** `2026-11-14`

## Approval record

| Field | Value |
| --- | --- |
| Status | **Approved** |
| Decision | Approved (verbally by the instructor in class) |
| Conditions from instructor | None |

---

## 1. Task formulation

Single-label, 101-way classification of **recipe web items**. Each item is an image together with the text of the web page the image was found on. The model predicts which of 101 dish categories the item shows.

- **Input:** one RGB image (224 × 224 for the image encoder) and the title of the web page it was found on (WordPiece tokens, truncated to 64 tokens).
- **Output:** logits over 101 classes, trained with cross-entropy.
- **Why multimodal:** the two modalities carry different evidence.
  - The image shows appearance: plating, texture, colour. In A2, appearance alone reached 86 % macro-F1 on the clean Food-101 photos.
  - The page title names the dish in most items (73 % contain the class name, section 2.4). It also names variants that change the appearance ("Apple Pie Moonshine", "Apple Pie Egg Rolls") and often the website.
  - Each modality fails on its own in different ways. Web images can be ads, step-by-step photos or pictures shared by two dishes. Titles can be generic ("Recipes", "Ingredients"), describe a derived dish, or name another class. These are the support and conflict cases required by section 6.11 of the spec.
- **Continuity with A2:** the label space is the same as Food-101 in A2, so the image pipeline (data loader, pHash duplicate search, ResNet-50 fine-tuning recipe, evaluation and decision rule) carries over. The new work is the text modality, cross-modal alignment and fusion.

## 2. Dataset

### 2.1 Source

**UPMC Food-101**: X. Wang, D. Kumar, N. Thome, M. Cord, F. Precioso, *"Recipe recognition with large multimodal food dataset"*, ICME Workshops 2015 (doi 10.1109/ICMEW.2015.7169757).

- **Collection:** the authors searched Google Images for each of the 101 Food-101 category names followed by "recipe". They kept each image together with the HTML page it came from.
- **Twin of A2's dataset:** it has the same 101 classes as ETHZ Food-101 (A2), but different images. ETHZ images come from foodspotting.com; UPMC images come from Google Image Search.
- **Distribution:** the original download link in the paper has reportedly expired. We use the Kaggle mirror `gianmarco96/upmcfood101`, which can be attached to Kaggle notebooks with the Internet off.
- **Layout of the mirror** (verified by our inventory run, `scripts/inventory_upmc.py`, Kaggle notebook `bathanhphan/co3133-a3-inventory`, 2026-11-14):
  - `images/{train,test}/<label>/<label>_<num>.jpg`: 90,704 JPEG files, 7.70 GB.
  - `texts/{train,test}_titles.csv`: three columns, no header: image file name, **page title**, label.
  - The mirror has **only the page title**, not the cleaned full page text used by MMBT (`texts_txt/`). Our text modality is therefore the page title.
- **Licence:** research use only. We will not redistribute images or texts, only code, split ID lists and checkpoints.

### 2.2 Size and threshold check

All counts below are measured on the Kaggle mirror by our inventory run.

| Quantity | Measured | Spec threshold (6.5) | Met? |
| --- | ---: | ---: | :---: |
| Classes | 101 (same set in images and CSVs) | ≥ 5 | Yes |
| Image files: train / test | 67,988 / 22,716 | - | - |
| Title rows: train / test | 67,972 / 22,716 | - | - |
| Image-title pairs matched by file name: train / test | 67,972 / 22,716 (no unmatched title rows, no label mismatch) | - | - |
| Train images without a title row | 16 (all `breakfast_burrito`) | - | - |
| Items per class: train / test | 591-713 (mean 673) / 197-238 (mean 225) | - | - |
| Images failing to decode / near-constant images | 0 / 3 | - | - |
| **Our training pairs after a 5,000-item val hold-out** | **≈ 62,970** | ≥ 5,000 paired samples | Yes (≈ 12×) |

- **Missing modality is rare in the raw data.** Only 16 of 90,704 items (0.02 %) have no title, and no title lacks an image. Missing-modality handling (5.1 item 5, 5.4) is therefore evaluated mainly by **simulated** removal and on the subset of uninformative titles (2.3, criterion 2). It is not evaluated on a large natural missing set.
- **Mildly imbalanced:** the max/min class ratio is 1.21 in train. The smallest classes are `lobster_bisque`, `escargots`, `hot_and_sour_soup` and `fish_and_chips`.

**Reference numbers** from Kiela et al., *Supervised Multimodal Bitransformers*, 2019 (test accuracy, mean ± std):
- image-only: 63.2 ± 0.6
- BERT text-only: 87.2 ± 0.1
- ConcatBERT: 90.0 ± 0.6
- late fusion: 91.1 ± 0.1
- MMBT: 92.1 ± 0.1

These numbers use the **full cleaned page text**, not titles, and a different split. They show that text dominates on this dataset, but they are not directly comparable with ours. Our titles are much shorter (median 7 words), so we expect a smaller text-image gap. A rule that predicts the class whose name appears in the title already reaches 71 % accuracy (section 2.4).

### 2.3 Modalities and pair validity

| Modality | Content | Encoder input |
| --- | --- | --- |
| A - Image | one RGB JPEG per item, web resolution (varies) | resized, normalised 224 × 224 crop |
| B - Text | title of the same web page. Measured: median 7 words / 48 characters, p99 20 words, max 135 words; contains HTML entities (`&#039;`, `&#8211;`) and often a site name ("\| Taste of Home", "on Pinterest") | HTML-unescaped, lower-cased WordPiece tokens, max 64 |

**Linking entity:** the **web page**. The image and the text of one item were extracted from the same crawled page, and they share the file stem `<label>_<num>`. Pairs are built only by matching this ID. We never pair items at random and never pair across datasets.

**Pair validity criteria**, checked by `scripts/build_pairs.py` and reported in the EDA:

1. **Image is valid:** the file decodes with PIL, the shorter side is ≥ 64 px, and it is not a single-colour placeholder (pixel std > 2). Items failing this are *image-missing*. Measured: 3 items (all images decode, the smallest side is 120 px).
2. **Text is valid:** after HTML unescaping and cleaning, the title contains at least one **content word**. A content word has ≥ 3 letters and is not on a short list of generic web words (`recipe(s)`, `ingredients`, `food`, `home`, `blog`, `pinterest`, ...; the list is fixed on train only and published with the code). Items failing this are *text-missing*.
   - A word-count threshold would be wrong for titles: "Chocolate Samosas" is short but informative.
   - Upper bound from the inventory: 819 train and 285 test titles have a single word (e.g. "Recipes", "Ingredients", "Perfect").
3. **Same item:** image and text share the same `<label>_<num>` file name and the same class. Measured: every title row matches exactly one image in the same split and class.
4. **Pair is semantically valid (manual audit):** we review 200 random train pairs (10 pairs from each of 20 random classes). Each pair is labelled as one of:
   - valid pair
   - image unrelated to the dish (logo, ad, person, collage)
   - text unrelated to the dish (boilerplate, another recipe)
   - wrong class label

   We report the rate of each category with a Wilson 95 % confidence interval (the `noise_audit.py` method from A2).

A **complete pair** satisfies criteria 1-3. The main metrics use complete pairs only, which keeps them comparable with the literature. Items missing one modality form a separate missing-modality evaluation set. Items missing both modalities are dropped, and we report how many there are.

### 2.4 Known properties and biases

Measured by the inventory run; the M2 EDA adds the rest.

1. **Label leakage through the query:** the images were retrieved with "<class name> recipe", so the title often contains the class name itself.
   - 73.0 % of train titles and 73.2 % of test titles contain their own class name (exact match after replacing `_` with a space). Plurals and synonyms will be added in the EDA.
   - Predicting the most-mentioned class name gives **71.1 % train / 71.2 % test accuracy without any learning**. 25.8 % of titles contain no class name at all.
   - 4.2 % of titles contain the name of *another* class, e.g. "Apple Pie Egg Rolls" or a cake named after another dish. These are candidate conflict cases.
   - Ablation E2 targets this leakage.
2. **Web noise in both modalities:** the authors did not clean the crawl. The pair audit (2.3) measures how often each modality is uninformative.
3. **Exact duplicate images** (identical bytes, md5): measured before any near-duplicate search.
   - 2,643 groups containing 6,240 images; the largest group has 16 copies.
   - **1,253 test images (5.5 %) have an exact copy in train.** For 157 of them, the train copy carries a different label only.
   - 437 groups (942 images) span two or more labels, e.g. the same photo under `filet_mignon` and `steak`. This is label ambiguity built into the crawl.

   Handling is in section 3.3.
4. **Repeated titles:** 10,861 titles occur more than once (4,874 across train/test, 801 across labels), e.g. "Miso Soup Recipe on Pinterest". They are short and generic, so an identical title does **not** mean the same page. Section 3.3 explains how this is treated.
5. **Site names inside titles:** many titles end with the site name ("\| Taste of Home", "\| iFood.tv", "on Pinterest"). This can act as a shortcut feature, and it is the only available proxy for the source website. The EDA extracts the trailing site segment and reports its distribution per class.
6. **Class balance:** mild imbalance, max/min ratio 1.21 (train) and 1.21 (test). We report macro-F1 as the primary metric.
7. **Text length:** median 7 words (p95 15, p99 20, max 135). The EDA reports the DistilBERT token distribution and the truncation rate at 64 tokens.
8. **Image geometry:** all images are RGB JPEG.
   - Shorter side: median 387 px, range 120-900 px.
   - Aspect ratio w/h: median 1.33. Extreme values (0.09 and 8.73) come from banners and collages.
   - Pre-resizing to a shorter side of 256 px therefore keeps almost all detail used at 224 px.
9. **Language and culture:** no title is mostly non-ASCII. Pages are mostly English and from Western recipe sites, the same cultural bias as in A2.
10. **Pretraining overlap:**
   - ImageNet-1k contains some food classes (see A2).
   - DistilBERT was pretrained on Wikipedia and BookCorpus, which describe many of these dishes.
   - Gains from pretrained encoders therefore partly reflect prior exposure to the classes.

### 2.5 Planned EDA (task-appropriate, spec 6.10)

- **Pair counts:** per class and per split. Number of complete, text-missing, image-missing and dropped items.
- **Pair quality:**
  - results of the manual 200-pair audit, with confidence intervals
  - share of titles that mention their own class name (with plural and synonym variants), per class
  - share of titles that mention a *different* class name more often than their own (a textual conflict indicator)
- **Missing-modality rates:** per class and per split, under criteria 1-2.
- **Image distributions:** width, height and aspect ratio; per-channel mean/std; file size; sample grid per class. These reuse A2's `eda.py`.
- **Text distributions:**
  - word count and DistilBERT token count (histogram, truncation rate at 64 tokens)
  - top TF-IDF terms per class; vocabulary size
  - distribution of trailing site names
- **Duplicates:** exact (md5) and near-duplicate (pHash) image clusters and repeated titles, both within and across splits and classes (section 3.3).
- **Batch example:** a printed batch with image tensor shape, `input_ids` / `attention_mask` shapes, decoded text snippets and labels.

## 3. Splitting plan

### 3.1 Splits

| Split | Source | Size |
| --- | --- | ---: |
| Train | official train pairs minus validation | ≈ 62,970 |
| Validation | 5,000 items from official train, stratified by class, grouped by image-duplicate cluster | ≈ 5,000 |
| Test | official test list (untouched) | 22,716 |
| Test, deduplicated | test minus items whose image (near-)duplicates a train image | ≤ 21,463 (≥ 1,253 removed by md5 alone) |

- **Split unit:** the **item** (one web page = one image + one title). The page is the entity that links the two modalities, so an image and its title can never fall into different splits.
- **Grouping:** the mirror has no URL or site ID; the only site proxy is the site name inside some titles (2.4 item 5). The grouping we can apply reliably is the **image-duplicate cluster**: items are connected when their images are identical (md5) or near-duplicates (pHash, section 3.3). All items of one cluster stay in the same split for train/val, so the same picture can never be in both train and val.
- **Stratification:** `sklearn.model_selection.StratifiedGroupKFold(n_splits=round(N_train / 5000), shuffle=True, random_state=42)` on the official train items, with class as the stratum and cluster ID as the group. We take one fold as validation (≈ 5,000 items) and the rest as train.
- **Seeds:** the split seed is fixed at **42**, the same convention as A1 and A2. Training seeds are 42, 123 and 2026.
- **Committed files:** the split is written once to `data/splits/{train,val,test}.csv` (columns `id, label, img_path, has_img, has_txt, cluster_id, test_dup`) and committed, so every model uses byte-identical splits. Titles are not redistributed; they are looked up by `id` in the Kaggle CSVs.

### 3.2 Use of each split

- **Train:** gradient updates only.
- **Validation:** hyper-parameters, late-fusion weight, early stopping and checkpoint selection (best validation macro-F1).
- **Test:** evaluated **once per final checkpoint** and never used for any decision. The same holds for the deduplicated test set and the simulated missing-modality variants.

### 3.3 Leakage prevention (cross-modal)

1. **Image duplicates:**
   - Exact copies are linked by md5. The inventory found 2,643 groups.
   - Near-duplicates are linked by a 64-bit pHash with Hamming distance ≤ 4, reusing `src/data/phash.py` from A2. These catch re-encoded or resized copies.
2. **Titles are not used to link items on their own.** Identical titles are common and generic (2.4 item 4), so linking on the title alone would merge thousands of unrelated pages into a few giant clusters. Two items are treated as the same page only when their images are linked **and** their normalised titles are identical. This is reported, and it does not change the clusters.
3. **Clusters:** clusters are the connected components of the image links. Within the official train list they are kept intact by the grouped split (3.1).
4. **Train-test duplicates:** the official test list stays fixed for comparability, so we cannot move clusters across that boundary. Instead, test items whose image is linked to any train image are flagged: at least 1,253 items (5.5 %) from md5 alone. We report all test metrics twice:
   - on the full test set
   - on the **deduplicated test set**, with flagged items removed (number reported)
5. **Cross-class duplicates:** 437 image groups (942 images) carry two or more labels. Those items are listed, kept in train (they are real label noise) and excluded from the deduplicated test set. They are also used as examples of conflict and label noise, since the same picture is paired with different titles and labels.
6. **Statistics:** text-side statistics (vocabulary, TF-IDF idf, generic-word list of criterion 2, class-name lexicon used in E2) are fitted on **train only**. Image models use ImageNet mean/std.
7. **ETHZ Food-101 overlap** (only if optional E3 uses the A2 checkpoint): before any run that starts from the A2 Food-101 checkpoint, we pHash-match UPMC test images against all ETHZ Food-101 images and exclude the matches from E3's test set.

## 4. Metrics

| Metric | Role |
| --- | --- |
| Macro-F1 | **Primary**: model selection and the decision rule |
| Top-1 accuracy | Mandatory; comparable with the literature |
| Top-5 accuracy, per-class F1, confusion matrix (top confused pairs) | Error analysis |
| Expected calibration error (15 bins) | Secondary; fusion can be over-confident |
| Accuracy / macro-F1 on full vs deduplicated test, and under simulated modality removal | Leakage check, missing-modality handling |
| Modality agreement breakdown (section 5.5) | Support/conflict analysis |
| Parameters, training time (GPU-h), inference throughput (pairs/s, batch 128), latency (ms, batch 1) | Compute cost |

**Decision rule (same as A1/A2):** method A is declared better than B only if both hold:
- the difference in mean test macro-F1 over 3 seeds exceeds the larger of the two seed standard deviations, **and**
- a McNemar test on seed 42 rejects equal error rates at p < 0.05.

A multimodal model that fails to beat the best unimodal model under this rule is reported as such and analysed (spec 6.6).

## 5. Model plan

All encoders are pretrained. Every model shares the same split, the same image transforms and the same text tokenisation.

| ID | Model | Modalities | Params (≈) | Role | Status |
| --- | --- | --- | ---: | --- | --- |
| UA | **Image-only**: ResNet-50 (torchvision `IMAGENET1K_V2`), new 101-way head. Same fine-tuning recipe as A2 P1 | A | 23.7 M | Unimodal baseline A | `[MANDATORY]` |
| UB | **Text-only**: DistilBERT-base-uncased, `[CLS]` → dropout → linear 101 | B | 66.4 M | Unimodal baseline B | `[MANDATORY]` |
| K0 | Keyword rule: predict the most-mentioned class name in the title. No training; measured at 71.2 % test accuracy (2.4) | B | 0 | Leakage floor for every text model | reported |
| T0 | TF-IDF (word 1-2 grams + char 3-5 grams) + logistic regression on titles, CPU only | B | - | Cheap text reference; measures how much a bag of words already solves | `[OPTIONAL]` |
| F0 | **Late fusion**: weighted average of UA and UB softmax outputs, weight w tuned on val. No training | A + B | 0 extra | Zero-cost fusion reference | reported alongside F1 |
| F1 | **Concat fusion (intermediate, simple)**: pooled image feature (2048) and text `[CLS]` (768), each projected to 512 → concat → MLP (1024 → 512 → 101), trained end-to-end | A + B | ≈ 91.6 M | Simple fusion baseline | `[MANDATORY]` |
| F2 | **Cross-modal transformer fusion** with auxiliary unimodal heads and modality dropout (5.1) | A + B | ≈ 97 M | Improved fusion design | `[MANDATORY]` |

Exact parameter counts will be printed by the code and reported in M2.

### 5.1 Improved fusion F2 (design)

1. **Modality-specific encoders and representations:**
   - The ResNet-50 `layer4` feature map (7 × 7 × 2048) becomes **49 image tokens**, linearly projected to d = 512, plus 2-D learned position embeddings.
   - DistilBERT's last hidden states give **≤ 64 text tokens**, projected 768 → 512.
   - Learned modality-type embeddings are added to both.
2. **Alignment:** alignment happens at the item level, through the shared page ID, and at the token level, learned by attention. There is no word-region annotation, so token-level alignment is not supervised; we inspect it qualitatively through attention maps.
3. **Fusion stage and operation (intermediate):**
   - A learned `[FUS]` token is prepended to [image tokens; text tokens].
   - The sequence goes through 2 transformer encoder layers (8 heads, d = 512, FFN 2048, pre-LN, dropout 0.1), with text padding masked.
   - The output at `[FUS]` → linear 101.

   Unlike F1, every image region can attend to every text token, and the reverse. Conflicts can therefore be resolved per sample rather than by a fixed weight.
4. **Losses:**
   - Overall loss: L = CE(fused) + λ · [CE(image head) + CE(text head)], with λ = 0.5.
   - The image head and text head are linear heads on the pooled image feature and the text `[CLS]`.
   - **Motivation:** the title often contains the label (2.4), so the text branch can fit the training set quickly. Without per-modality losses, joint training tends to under-train the image branch ("modality laziness"). The auxiliary heads keep each encoder discriminative on its own, which also supports missing-modality inference.
5. **Missing-modality handling:**
   - **Modality dropout during training:** with p = 0.15, one modality's tokens are replaced by a single learned `[MISSING_IMG]` / `[MISSING_TXT]` token, never both at once.
   - **At inference:** items failing a validity criterion (2.3) receive the same token: uninformative titles, the 16 train images without a title, and the 3 near-constant images.
   - F1 handles a missing modality by zero-filling its projected vector. F0 falls back to the available unimodal model.
6. **Pretraining and fine-tuning strategy:**
   - Encoders start from ImageNet-1k (ResNet-50) and Wikipedia/BookCorpus (DistilBERT) pretraining.
   - **Stage 1 (1 epoch):** both encoders frozen; only the projections, fusion layers and heads are trained.
   - **Stage 2:** full fine-tuning with discriminative learning rates.

### 5.2 Training setup (initial plan; tuned on validation only)

- **Image transforms:** `RandomResizedCrop(224, scale=(0.35, 1.0))`, horizontal flip and light colour jitter for training; `Resize(256)` + `CenterCrop(224)` for val/test. This is the A2 setting, and it is identical across UA, F1 and F2.
- **Text:**
  - Cleaning: `html.unescape`, lower-case, collapse whitespace. Site names are kept in the main runs; the EDA measures whether they act as shortcuts.
  - Tokenisation: DistilBERT uncased WordPiece, truncation to 64 tokens. Titles have p99 = 20 words, so we expect < 0.5 % truncation; the EDA reports the exact rate.
  - No text augmentation.
- **Optimiser:** AdamW, weight decay 0.01 for text parameters and 0.05 for image parameters.
- **Learning rates:**
  - UA: backbone 1e-4, head 1e-3 (A2)
  - UB: 3e-5 for the encoder, 1e-3 for the head
  - F1/F2: image encoder 1e-4, text encoder 3e-5, fusion layers and heads 5e-4
- **Schedule and precision:** linear warm-up over 5 % of steps, then cosine decay. bf16 autocast on the RTX PRO 6000; fp16 with gradient scaling on a T4 fallback.
- **Epochs and batch size:**
  - UA, F1, F2: 10 epochs
  - UB: 5 epochs (BERT-style fine-tuning converges faster)
  - Batch size 64 for all.
- **Checkpoint selection:** best validation macro-F1, with early stopping after 3 epochs without improvement. A checkpoint is saved every epoch, so runs can resume after a session ends.

### 5.3 Experiments and ablations

Each experiment changes **one** factor. The split, transforms, tokenisation, epochs, optimiser and seeds stay fixed.

| ID | Hypothesis | Changed factor | Fixed | Decision criterion | Status |
| --- | --- | --- | --- | --- | --- |
| E0 | UB beats the keyword rule K0 (71 %) and beats UA, and both fusions beat the best unimodal model. With titles instead of full text, the text-image gap is smaller than the literature's 24 pp. | Modality set: K0 vs UA vs UB vs F0 vs F1 vs F2 | everything else | Decision rule (section 4) | `[MANDATORY]` main comparison |
| E1 | **Fusion component.** The auxiliary unimodal losses make F2 better and more robust:<br>(a) ≥ 0.5 pp test macro-F1;<br>(b) higher image-head accuracy inside the fused model;<br>(c) smaller drop when text is removed at test time. | F2 with λ = 0.5 vs λ = 0 | everything else | Decision rule for (a); (b) and (c) are reported with mean ± std | `[MANDATORY]` |
| E2 | **Modality contribution.** Much of the text signal is the class name injected by the search query (73 % of titles). Masking class-name mentions:<br>• lowers UB macro-F1 by ≥ 10 pp;<br>• increases the fusion gain over UB (F2 − UB);<br>• raises the share of correct F2 decisions on conflict samples (5.5). | Text as is vs text with every class-name mention (train-fitted lexicon of the 101 names + variants) replaced by `[MASK]`, in train **and** test, for UB, F1 and F2 | images, everything else | Decision rule for each model; F2 − UB gap with mean ± std | `[MANDATORY]` |
| E-mod | **Test-time modality removal** (no training). We evaluate F1 and F2 with the image replaced by the missing token / zero vector, then with the text replaced. This measures how much each trained fusion model relies on each modality. | Modality available at test time | trained checkpoints | Reported table | `[MANDATORY]`, evaluation only |
| E3 | Initialising the image encoder from the A2 ETHZ Food-101 checkpoint (P1) instead of ImageNet raises UA by ≥ 3 pp. The gain in F2 is smaller, because text already carries the class information. | ImageNet vs A2 Food-101 init (after the ETHZ overlap check, 3.3 item 7) | everything else | Decision rule | `[OPTIONAL]` |
| E4 | Modality dropout improves accuracy under simulated text removal (0-100 % of test items) and on uninformative titles, at ≤ 0.5 pp cost on complete pairs. | F2 with p = 0.15 vs p = 0 | everything else | Robustness curve + decision rule on complete pairs | `[OPTIONAL]` |

### 5.4 Missing-modality evaluation

Natural missingness is almost absent (16 of 90,704 items, 2.2), so UA, UB, F0, F1 and F2 are evaluated on:
- complete test pairs with one modality removed for 25 / 50 / 100 % of the items (seed 42), using each model's fallback (5.1 item 5)
- the test subset whose title fails criterion 2 (uninformative titles), as the closest natural case

### 5.5 Support / conflict analysis plan (spec 6.11)

1. **Grouping:** on the test set, each item is grouped by the predictions of the two unimodal models (seed 42):

   | Group | UA (image) | UB (text) | Meaning |
   | --- | --- | --- | --- |
   | Support | correct | correct | modalities agree on the true class |
   | Conflict-I | correct | wrong | image right, text misleading |
   | Conflict-T | wrong | correct | text right, image misleading |
   | Both wrong | wrong | wrong | — |

   "Both wrong" is split further into agreeing-wrong and disagreeing-wrong.
2. **Per-group results:** for each group we report its size, the accuracy of F0/F1/F2, and which modality the fused prediction followed.
3. **Expectation:** we expect the fusion gain to come almost entirely from the Conflict groups. We check this claim directly.
4. **Qualitative evidence:** ≥ 5 examples per group, each showing:
   - the image with a Grad-CAM map for the image branch (reusing A2's `gradcam.py`)
   - the title with the top attended tokens from `[FUS]` (F2 attention)
   - the UA, UB and F2 predictions
5. **Error taxonomy for conflicts:**
   - image unrelated to the dish
   - title is generic (site name only, "Recipes")
   - title names a derived dish or another class ("Apple Pie Moonshine")
   - visually similar classes
   - wrong label (cross-class duplicate images)

   Rates are estimated from a manual review of 100 conflict samples.
6. **Class-name breakdown:** the analysis is repeated separately for titles that contain the class name (73 %) and titles that do not (27 %).

## 6. Compute estimate

**Hardware:**
- **Primary:** Kaggle notebook, NVIDIA RTX PRO 6000 Blackwell Server Edition (95 GB), 48 vCPU, **Internet off**. This is the same environment used for every reported A2 run.
- **Fallback:** Kaggle T4 (16 GB), or a teammate's RTX 3060 (12 GB) for debugging.
- No training is done on the laptop.

**Offline workflow (Internet off):** extends A2's `scripts/make_kaggle_bundle.py`.
- The **code bundle** is a private Kaggle dataset containing:
  - the repo at a tagged commit (+ a `GIT_COMMIT` file)
  - torchvision ResNet-50 `IMAGENET1K_V2` weights
  - a `distilbert-base-uncased` snapshot (weights + tokenizer files, loaded with `local_files_only=True`)
  - wheels for any package missing from the Kaggle image (versions recorded in `env.json`)
- The **data:** the `upmcfood101` Kaggle dataset (7.70 GB) is attached as input. Images are pre-resized once to a shorter side of 256 px in a CPU-only notebook (no GPU quota), and the titles are pre-tokenised once into a `.npy` cache. Both are stored as a second private dataset.
  - CPU cost reference: the inventory run decoded all 90,704 images in 25 min on the 4-vCPU CPU notebook. Pre-resizing has the same order of cost, so it should take < 1 h.

**Throughput assumptions** (to be replaced by a measured 1-epoch timing run before any full run):
- A2 measured ResNet-50 fine-tuning at ≈ 1,900 img/s on this GPU, limited by data loading.
- Titles are short (≤ 64 tokens), so the text encoder is cheap. With images pre-resized and titles pre-tokenised, we assume:
  - UA: ≈ 1,800 pairs/s
  - UB: ≈ 4,000 seq/s
  - F1: ≈ 1,500 pairs/s
  - F2: ≈ 1,300 pairs/s
- One epoch is ≈ 63 k pairs (train) + 5 k (val).

| Runs | min/epoch | Epochs | Seeds | Est. GPU-h |
| --- | ---: | ---: | ---: | ---: |
| 1-epoch timing runs (UA, UB, F1, F2) | - | 1 | 1 | 0.2 |
| UA image-only | 0.6 | 10 | 3 | 0.3 |
| UB text-only | 0.3 | 5 | 3 | 0.1 |
| F1 concat fusion | 0.75 | 10 | 3 | 0.4 |
| F2 cross-modal fusion | 0.9 | 10 | 3 | 0.5 |
| E1 F2 without aux losses | 0.9 | 10 | 3 | 0.5 |
| E2 masked class names: UB + F1 + F2 | - | 5/10/10 | 3 | 1.0 |
| E-mod, missing-modality and test evaluation of all runs | - | - | - | 0.3 |
| **Mandatory total** | | | | **≈ 3.3** |
| E3 A2-checkpoint init (UA + F2) | - | 10 | 3 | 0.8 |
| E4 F2 without modality dropout | 0.9 | 10 | 3 | 0.5 |
| T0 TF-IDF + LR (CPU notebook) | - | - | 1 | 0 GPU |
| **Total including optional** | | | | **≈ 4.6** |

- **Headroom:** the estimate is ≈ 5 GPU-h, which leaves > 5× headroom within one account's weekly quota for estimation error and re-runs. No single run exceeds ≈ 10 min.
- **T4 fallback:** assuming ≈ 5× slower, the mandatory set takes ≈ 17 GPU-h, which still fits one account's weekly quota.
- **Storage:**
  - the raw dataset (7.70 GB) is read from Kaggle input, not copied
  - ≈ 2.5 GB after pre-resizing (estimate)
  - checkpoints are ≈ 95 MB (UA), ≈ 255 MB (UB), ≈ 360-390 MB (F1/F2); only the selected checkpoint per configuration and seed is kept
- **Fallback if throughput is lower than assumed:** drop optional items in the order E3 → E4 → T0, then train E2 with seed 42 only.

## 7. Deliverables timeline

| Milestone | Content |
| --- | --- |
| M1 (this document) | Dataset proposal: modalities, pair validity, split unit, leakage prevention, metrics, baselines, compute estimate |
| M2 | Pairs built and audited, EDA complete, split files committed; UA, UB, F0 and F1 trained with quantitative results |
| M3 | F2, E1, E2, E-mod (and optional E3/E4/T0); support/conflict analysis, error analysis, compute cost; report, slides, video, pages, checkpoints |

## 8. Risks and limitations

- **No URL or site ID in the release.** Grouping uses image-duplicate clusters, not source websites. Pages from the same site share title templates, and site names appear inside titles, so some site-level leakage and shortcut learning may remain. The site-name analysis in the EDA measures this, and it will be stated as a limitation.
- **Titles only, not full page text.** The Kaggle mirror is the only available source and carries titles only. Results are therefore not directly comparable with MMBT-style numbers on full text (2.2). The text modality is short but strongly label-bearing: the keyword rule reaches 71 %.
- **Text leakage and fusion gains.** Fusion gains over UB may be small. E2 is designed so that the image contribution is still measurable. If fusion does not beat UB under the decision rule, we analyse why, as the spec requires.
- **Official test list is fixed.** At least 5.5 % of test images are exact copies of train images and cannot be regrouped. We report results on both the full and the deduplicated test sets.
- **Noisy labels and noisy pairs** in train and test (web crawl, not manually reviewed). Their rate is estimated by the audit. Test metrics are therefore an upper bound on how clean evaluation can be.
- **Compute scope.** We use mid-sized encoders (ResNet-50, DistilBERT). Large vision-language models (CLIP ViT-L, BLIP) are out of scope for the mandatory part.

## References

1. X. Wang, D. Kumar, N. Thome, M. Cord, F. Precioso. *Recipe recognition with large multimodal food dataset.* ICME Workshops, 2015.
2. D. Kiela, S. Bhooshan, H. Firooz, E. Perez, D. Testuggine. *Supervised Multimodal Bitransformers for Classifying Images and Text.* arXiv:1909.02950, 2019. Code: `facebookresearch/mmbt`.
3. L. Bossard, M. Guillaumin, L. Van Gool. *Food-101 – Mining Discriminative Components with Random Forests.* ECCV, 2014.
4. V. Sanh, L. Debut, J. Chaumond, T. Wolf. *DistilBERT, a distilled version of BERT.* arXiv:1910.01108, 2019.
5. K. He, X. Zhang, S. Ren, J. Sun. *Deep Residual Learning for Image Recognition.* CVPR, 2016.
