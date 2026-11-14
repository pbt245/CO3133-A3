# Assignment 3 - Dataset Proposal (Milestone M1)

**Course:** Deep Learning and Its Applications (CO3133), Semester 261 - Instructor: Lê Thành Sách
**Group:** G-roundhog - Phan Bá Thanh (2353084), Trần Công Hoàng Phước (2352966)
**Task option:** Image-text classification (multimodal)
**Submitted:** `[YYYY-MM-DD]`

## Approval record

| Field | Value |
| --- | --- |
| Status | `[pending]` |
| Decision (Approved / Approved with conditions / Rejected) | `[ ]` |
| Conditions from instructor | `[ ]` |
| Decision date | `[YYYY-MM-DD]` |

---

## 1. Task formulation

Single-label, 101-way classification of **recipe web items**. Each item is an image together with the text of the web page the image was found on. The model predicts which of 101 dish categories the item shows.

- **Input:** one RGB image (224 × 224 for the image encoder) and its paired recipe text (WordPiece tokens, truncated to 256 tokens).
- **Output:** logits over 101 classes, trained with cross-entropy.
- **Why multimodal:** the two modalities carry different evidence.
  - The image shows appearance: plating, texture, colour. In A2, appearance alone reached 86 % macro-F1 on the clean Food-101 photos.
  - The text lists ingredients, cooking steps and often the dish name.
  - Each modality fails on its own in different ways. Web images can be ads, step-by-step photos or unrelated pictures. Web text can be page boilerplate, a different recipe, or a page that covers several dishes. These are the support and conflict cases required by section 6.11 of the spec.
- **Continuity with A2:** the label space is the same as Food-101 in A2, so the image pipeline (data loader, pHash duplicate search, ResNet-50 fine-tuning recipe, evaluation and decision rule) carries over. The new work is the text modality, cross-modal alignment and fusion.

## 2. Dataset

### 2.1 Source

**UPMC Food-101**: X. Wang, D. Kumar, N. Thome, M. Cord, F. Precioso, *"Recipe recognition with large multimodal food dataset"*, ICME Workshops 2015 (doi 10.1109/ICMEW.2015.7169757).

- **Collection:** the authors searched Google Images for each of the 101 Food-101 category names followed by "recipe". They kept each image together with the HTML page it came from, and later cleaned the page into plain recipe text.
- **Twin of A2's dataset:** it has the same 101 classes as ETHZ Food-101 (A2), but different images. ETHZ images come from foodspotting.com; UPMC images come from Google Image Search.
- **Distribution:** the original download link in the paper has reportedly expired. We will use the Kaggle mirror `gianmarco96/upmcfood101`, which can be attached to Kaggle notebooks with the Internet off. We also follow the file conventions of the MMBT reference preparation script (`facebookresearch/mmbt`, `scripts/food_101.py`): images under `images/{train,test}/<label>/<label>_<num>.jpg` and texts under `texts_txt/<label>/<label>_<num>.txt`.
- **Licence:** research use only. We will not redistribute images or texts, only code, split ID lists and checkpoints.

### 2.2 Size and threshold check

The counts below come from the literature. Our EDA will re-count them from the files we download (section 2.4), and the M2 report will state any differences.

| Quantity | Value (literature) | Spec threshold (6.5) | Met? |
| --- | ---: | ---: | :---: |
| Classes | 101 | ≥ 5 | Yes |
| Total image-text items | ≈ 90,700 (90,688 reported in multiple papers) | - | - |
| Official train items | ≈ 67,970 | - | - |
| Official test items | ≈ 22,716 | - | - |
| Items with both modalities after MMBT cleaning (train / dev / test) | 60,101 / 5,000 / 21,695 | - | - |
| **Our training pairs after val hold-out (expected)** | **≈ 60,000** | ≥ 5,000 paired samples | Yes (≈ 12×) |

The gap between the raw counts and the MMBT counts (≈ 2,900 train and ≈ 1,000 test items) is items whose text file is missing or empty. That is a **natural missing-text rate of about 4-5 %**, and we use these items for the missing-modality evaluation (section 5.4) instead of discarding them.

**Reference numbers** from Kiela et al., *Supervised Multimodal Bitransformers*, 2019 (test accuracy, mean ± std):
- image-only: 63.2 ± 0.6
- BERT text-only: 87.2 ± 0.1
- ConcatBERT: 90.0 ± 0.6
- late fusion: 91.1 ± 0.1
- MMBT: 92.1 ± 0.1

So text is the dominant modality on this dataset. That is a known risk, and it shapes our ablations (section 5.3).

### 2.3 Modalities and pair validity

| Modality | Content | Encoder input |
| --- | --- | --- |
| A - Image | one RGB JPEG per item, web resolution (varies) | resized, normalised 224 × 224 crop |
| B - Text | cleaned recipe text from the same web page: title, ingredients, instructions, page text | lower-cased WordPiece tokens, max 256 |

**Linking entity:** the **web page**. The image and the text of one item were extracted from the same crawled page, and they share the file stem `<label>_<num>`. Pairs are built only by matching this ID. We never pair items at random and never pair across datasets.

**Pair validity criteria**, checked by `scripts/build_pairs.py` and reported in the EDA:

1. **Image is valid:** the file decodes with PIL, the shorter side is ≥ 64 px, and it is not a single-colour placeholder (pixel std > 2). Items failing this are *image-missing*.
2. **Text is valid:** after cleaning, the text has ≥ 5 word tokens. Items failing this are *text-missing*.
3. **Same item:** image and text share the same `<label>_<num>` ID and the same class folder.
4. **Pair is semantically valid (manual audit):** we review 200 random train pairs (10 pairs from each of 20 random classes). Each pair is labelled as one of:
   - valid pair
   - image unrelated to the dish (logo, ad, person, collage)
   - text unrelated to the dish (boilerplate, another recipe)
   - wrong class label

   We report the rate of each category with a Wilson 95 % confidence interval (the `noise_audit.py` method from A2).

A **complete pair** satisfies criteria 1-3. The main metrics use complete pairs only, which keeps them comparable with the literature. Items missing one modality form a separate missing-modality evaluation set. Items missing both modalities are dropped, and we report how many there are.

### 2.4 Known properties and biases (to be quantified in the M2 EDA)

1. **Label leakage through the query:** the images were retrieved with "<class name> recipe", so the page text very often contains the class name itself. The EDA will report the share of texts containing the class name (and simple variants such as plural, no underscore, and common synonyms). This is the main reason text-only scores are high, and ablation E2 targets it.
2. **Web noise in both modalities:** the authors did not clean the crawl. The pair audit (2.3) measures how often each modality is uninformative.
3. **Duplicates:** the same image or the same page text can be returned for several queries, sometimes under two different classes (e.g. `beef_carpaccio` and `beef_tartare`). This creates both cross-split leakage and conflicting labels. Handling is in section 3.3.
4. **Class balance:** the number of items per class varies slightly, because some queries returned fewer usable pages. We report per-class counts per split and the max/min ratio.
5. **Text length:** web pages differ a lot in length, from a title only to a full blog post. We report the word and token length distributions per class and the share of texts truncated at 256 tokens.
6. **Language and culture:** pages are mostly English and from Western recipe sites, the same cultural bias as in A2. The EDA reports the share of non-English texts, estimated by the ASCII-letter ratio.
7. **Pretraining overlap:**
   - ImageNet-1k contains some food classes (see A2).
   - DistilBERT was pretrained on Wikipedia and BookCorpus, which describe many of these dishes.
   - Gains from pretrained encoders therefore partly reflect prior exposure to the classes.

### 2.5 Planned EDA (task-appropriate, spec 6.10)

- **Pair counts:** per class and per split. Number of complete, text-missing, image-missing and dropped items.
- **Pair quality:**
  - results of the manual 200-pair audit, with confidence intervals
  - share of texts that mention their own class name
  - share of texts that mention a *different* class name more often than their own (a textual conflict indicator)
- **Missing-modality rates:** per class and per split.
- **Image distributions:** width, height and aspect ratio; per-channel mean/std; file size; sample grid per class. These reuse A2's `eda.py`.
- **Text distributions:** word count and DistilBERT token count (histogram, truncation rate at 256 and 512 tokens); top TF-IDF terms per class; vocabulary size.
- **Duplicates:** image pHash clusters and text near-duplicate clusters, both within and across splits and classes (section 3.3).
- **Batch example:** a printed batch with image tensor shape, `input_ids` / `attention_mask` shapes, decoded text snippets and labels.

## 3. Splitting plan

### 3.1 Splits

| Split | Source | Size (expected, complete pairs) |
| --- | --- | ---: |
| Train | official train minus validation | ≈ 60,000 |
| Validation | 5,000 items from official train, stratified by class, grouped by duplicate cluster | 5,000 |
| Test | official test list (untouched) | ≈ 21,700 |
| Missing-modality eval | test items with exactly one modality missing | ≈ 1,000 |

- **Split unit:** the **item** (one web page = one image + one text). The page is the entity that links the two modalities, so an image and its text can never fall into different splits.
- **Grouping:** UPMC Food-101 has no URL or site ID in the released files. If the files we download do include a source URL, we will group by web domain instead and report this. Without it, the strongest grouping available is the **duplicate cluster**: we connect items whose images are near-duplicates **or** whose texts are near-duplicates (section 3.3). All items of one cluster stay in the same split for train/val, so the same page content can never be in both train and val.
- **Stratification:** `sklearn.model_selection.StratifiedGroupKFold(n_splits=round(N_train / 5000), shuffle=True, random_state=42)` on the official train items, with class as the stratum and cluster ID as the group. We take one fold as validation (≈ 5,000 items) and the rest as train.
- **Seeds:** the split seed is fixed at **42**, the same convention as A1 and A2. Training seeds are 42, 123 and 2026.
- **Committed files:** the split is written once to `data/splits/{train,val,test,test_missing}.csv` (columns `id, label, img_path, txt_path, has_img, has_txt, cluster_id`) and committed, so every model uses byte-identical splits.

### 3.2 Use of each split

- **Train:** gradient updates only.
- **Validation:** hyper-parameters, late-fusion weight, early stopping and checkpoint selection (best validation macro-F1).
- **Test:** evaluated **once per final checkpoint** and never used for any decision. The same holds for the missing-modality set.

### 3.3 Leakage prevention (cross-modal)

1. **Image near-duplicates:** compute a 64-bit pHash for every image, reusing `src/data/phash.py` from A2. Images with Hamming distance ≤ 4 are linked.
2. **Text near-duplicates:**
   - Normalise the text (lower-case, strip punctuation and whitespace), then link exact matches by SHA-1.
   - Also link near matches: MinHash over word 5-gram shingles with estimated Jaccard ≥ 0.9.
3. **Clusters:** clusters are the connected components of image links ∪ text links. Within the official train list they are kept intact by the grouped split (3.1).
4. **Train-test duplicates:** the official test list stays fixed for comparability, so we cannot move clusters across that boundary. Instead, test items that share a cluster with any train item are flagged. We report all test metrics twice:
   - on the full test set
   - on the **deduplicated test set**, with flagged items removed (number reported)
5. **Cross-class duplicates:** when one cluster has items with different labels, those items are listed. They are kept in train, since they are real label noise, but they are excluded from the deduplicated test set. They are also used as examples of conflict and label noise.
6. **Statistics:** text-side statistics (vocabulary, TF-IDF idf, class-name lexicon used in E2) are fitted on **train only**. Image models use ImageNet mean/std.
7. **ETHZ Food-101 overlap** (only if optional E3 uses the A2 checkpoint): before any run that starts from the A2 Food-101 checkpoint, we pHash-match UPMC test images against all ETHZ Food-101 images and exclude the matches from E3's test set.

## 4. Metrics

| Metric | Role |
| --- | --- |
| Macro-F1 | **Primary**: model selection and the decision rule |
| Top-1 accuracy | Mandatory; comparable with the literature |
| Top-5 accuracy, per-class F1, confusion matrix (top confused pairs) | Error analysis |
| Expected calibration error (15 bins) | Secondary; fusion can be over-confident |
| Accuracy / macro-F1 on full vs deduplicated test, and on the missing-modality set | Leakage check, missing-modality handling |
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
| T0 | TF-IDF (word 1-2 grams, 200 k features) + logistic regression, CPU only | B | - | Cheap text reference; measures how much a bag of words already solves | `[OPTIONAL]` |
| F0 | **Late fusion**: weighted average of UA and UB softmax outputs, weight w tuned on val. No training | A + B | 0 extra | Zero-cost fusion reference | reported alongside F1 |
| F1 | **Concat fusion (intermediate, simple)**: pooled image feature (2048) and text `[CLS]` (768), each projected to 512 → concat → MLP (1024 → 512 → 101), trained end-to-end | A + B | ≈ 91.6 M | Simple fusion baseline | `[MANDATORY]` |
| F2 | **Cross-modal transformer fusion** with auxiliary unimodal heads and modality dropout (5.1) | A + B | ≈ 97 M | Improved fusion design | `[MANDATORY]` |

Exact parameter counts will be printed by the code and reported in M2.

### 5.1 Improved fusion F2 (design)

1. **Modality-specific encoders and representations:**
   - The ResNet-50 `layer4` feature map (7 × 7 × 2048) becomes **49 image tokens**, linearly projected to d = 512, plus 2-D learned position embeddings.
   - DistilBERT's last hidden states give **≤ 256 text tokens**, projected 768 → 512.
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
   - **Motivation:** text dominates on this dataset (2.2). Without per-modality losses, joint training tends to under-train the weaker image branch ("modality laziness"). The auxiliary heads keep each encoder discriminative on its own, which also supports missing-modality inference.
5. **Missing-modality handling:**
   - **Modality dropout during training:** with p = 0.15, one modality's tokens are replaced by a single learned `[MISSING_IMG]` / `[MISSING_TXT]` token, never both at once.
   - **At inference:** naturally missing modalities (≈ 4-5 % of items) receive the same token.
   - F1 handles a missing modality by zero-filling its projected vector. F0 falls back to the available unimodal model.
6. **Pretraining and fine-tuning strategy:**
   - Encoders start from ImageNet-1k (ResNet-50) and Wikipedia/BookCorpus (DistilBERT) pretraining.
   - **Stage 1 (1 epoch):** both encoders frozen; only the projections, fusion layers and heads are trained.
   - **Stage 2:** full fine-tuning with discriminative learning rates.

### 5.2 Training setup (initial plan; tuned on validation only)

- **Image transforms:** `RandomResizedCrop(224, scale=(0.35, 1.0))`, horizontal flip and light colour jitter for training; `Resize(256)` + `CenterCrop(224)` for val/test. This is the A2 setting, and it is identical across UA, F1 and F2.
- **Text:**
  - Cleaning: lower-case, collapse whitespace, strip HTML remnants.
  - Tokenisation: DistilBERT uncased WordPiece, head truncation to 256 tokens. 256 vs 512 is decided from the EDA truncation rate and reported.
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
| E0 | Text-only beats image-only by ≥ 10 pp macro-F1, as in the literature (87.2 vs 63.2 accuracy), and both fusions beat the best unimodal model. | Modality set: UA vs UB vs F0 vs F1 vs F2 | everything else | Decision rule (section 4) | `[MANDATORY]` main comparison |
| E1 | **Fusion component.** The auxiliary unimodal losses make F2 better and more robust:<br>(a) ≥ 0.5 pp test macro-F1;<br>(b) higher image-head accuracy inside the fused model;<br>(c) smaller drop when text is removed at test time. | F2 with λ = 0.5 vs λ = 0 | everything else | Decision rule for (a); (b) and (c) are reported with mean ± std | `[MANDATORY]` |
| E2 | **Modality contribution.** Much of the text signal is the class name injected by the search query. Masking class-name mentions:<br>• lowers UB macro-F1 by ≥ 10 pp;<br>• increases the fusion gain over UB (F2 − UB);<br>• raises the share of correct F2 decisions on conflict samples (5.5). | Text as is vs text with every class-name mention (train-fitted lexicon of the 101 names + variants) replaced by `[MASK]`, in train **and** test, for UB, F1 and F2 | images, everything else | Decision rule for each model; F2 − UB gap with mean ± std | `[MANDATORY]` |
| E-mod | **Test-time modality removal** (no training). We evaluate F1 and F2 with the image replaced by the missing token / zero vector, then with the text replaced. This measures how much each trained fusion model relies on each modality. | Modality available at test time | trained checkpoints | Reported table | `[MANDATORY]`, evaluation only |
| E3 | Initialising the image encoder from the A2 ETHZ Food-101 checkpoint (P1) instead of ImageNet raises UA by ≥ 3 pp. The gain in F2 is smaller, because text already carries the class information. | ImageNet vs A2 Food-101 init (after the ETHZ overlap check, 3.3 item 7) | everything else | Decision rule | `[OPTIONAL]` |
| E4 | Modality dropout improves accuracy on the natural missing-text set and under simulated text removal (0-100 % of test items), at ≤ 0.5 pp cost on complete pairs. | F2 with p = 0.15 vs p = 0 | everything else | Robustness curve + decision rule on complete pairs | `[OPTIONAL]` |

### 5.4 Missing-modality evaluation

UA, UB, F0, F1 and F2 are all evaluated on two settings:
- the natural missing-modality set (3.1), using each model's fallback (5.1 item 5)
- complete test pairs with one modality removed for 25 / 50 / 100 % of the items (seed 42)

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
   - the text with the top attended tokens from `[FUS]` (F2 attention)
   - the UA, UB and F2 predictions
5. **Error taxonomy for conflicts:**
   - image unrelated to the dish
   - text is boilerplate or describes another recipe
   - text mentions several dishes
   - visually similar classes
   - wrong label (cross-class duplicates)

   Rates are estimated from a manual review of 100 conflict samples.
6. **Class-name breakdown:** the analysis is repeated separately for texts that contain the class name and texts that do not.

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
- The **data:** the `upmcfood101` Kaggle dataset is attached as input. Images are pre-resized once to a shorter side of 256 px in a CPU-only notebook (no GPU quota), and the texts are pre-tokenised once into a `.npy` cache. Both are stored as a second private dataset.

**Throughput assumptions** (to be replaced by a measured 1-epoch timing run before any full run):
- A2 measured ResNet-50 fine-tuning at ≈ 1,900 img/s on this GPU, limited by data loading.
- With images pre-resized and text pre-tokenised, we assume:
  - UA: ≈ 1,800 pairs/s
  - UB: ≈ 1,500 seq/s at 256 tokens
  - F1: ≈ 1,000 pairs/s
  - F2: ≈ 800 pairs/s
- One epoch is ≈ 60 k pairs (train) + 5 k (val).

| Runs | min/epoch | Epochs | Seeds | Est. GPU-h |
| --- | ---: | ---: | ---: | ---: |
| 1-epoch timing runs (UA, UB, F1, F2) | - | 1 | 1 | 0.2 |
| UA image-only | 0.6 | 10 | 3 | 0.3 |
| UB text-only | 0.7 | 5 | 3 | 0.2 |
| F1 concat fusion | 1.1 | 10 | 3 | 0.6 |
| F2 cross-modal fusion | 1.4 | 10 | 3 | 0.7 |
| E1 F2 without aux losses | 1.4 | 10 | 3 | 0.7 |
| E2 masked class names: UB + F1 + F2 | - | 5/10/10 | 3 | 1.5 |
| E-mod, missing-modality and test evaluation of all runs | - | - | - | 0.3 |
| **Mandatory total** | | | | **≈ 4.5** |
| E3 A2-checkpoint init (UA + F2) | - | 10 | 3 | 1.0 |
| E4 F2 without modality dropout | 1.4 | 10 | 3 | 0.7 |
| T0 TF-IDF + LR (CPU notebook) | - | - | 1 | 0 GPU |
| **Total including optional** | | | | **≈ 6.2** |

- **Headroom:** the estimate is ≈ 6 GPU-h, which leaves > 4× headroom within one account's weekly quota for estimation error and re-runs. No single run exceeds ≈ 15 min.
- **T4 fallback:** assuming ≈ 5× slower, the mandatory set takes ≈ 23 GPU-h, which still fits the two accounts' weekly quotas.
- **Storage:**
  - the raw dataset is read from Kaggle input (size recorded in the EDA)
  - ≈ 2 GB after pre-resizing
  - checkpoints are ≈ 95 MB (UA), ≈ 255 MB (UB), ≈ 360-390 MB (F1/F2); only the selected checkpoint per configuration and seed is kept
- **Fallback if throughput is lower than assumed:** drop optional items in the order E3 → E4 → T0, then train E2 with seed 42 only.

## 7. Deliverables timeline

| Milestone | Content |
| --- | --- |
| M1 (this document) | Dataset proposal: modalities, pair validity, split unit, leakage prevention, metrics, baselines, compute estimate |
| M2 | Pairs built and audited, EDA complete, split files committed; UA, UB, F0 and F1 trained with quantitative results |
| M3 | F2, E1, E2, E-mod (and optional E3/E4/T0); support/conflict analysis, error analysis, compute cost; report, slides, video, pages, checkpoints |

## 8. Risks and limitations

- **No URL or site ID in the release (expected).** Grouping uses duplicate clusters, not source websites. Pages from the same site share templates and boilerplate, and the 0.9-Jaccard threshold does not catch all of that. Some site-level leakage may remain; this will be stated as a limitation.
- **Text dominance.** Fusion gains over UB may be small (the literature shows about +3-5 pp accuracy). E2 is designed so that the image contribution is still measurable. If fusion does not beat UB under the decision rule, we analyse why, as the spec requires.
- **Mirror differences.** The Kaggle mirror may differ from the original release in counts, text field (full cleaned text vs page title only) or file layout. We verify this in the first CPU notebook, before M2. If the mirror only has titles, the task stays valid but the text becomes much shorter: we will report this, set the max length from the EDA, and keep E2 as the key analysis.
- **Official test list is fixed.** Train-test duplicates cannot be regrouped. We report results on both the full and the deduplicated test sets.
- **Noisy labels and noisy pairs** in train and test (web crawl, not manually reviewed). Their rate is estimated by the audit. Test metrics are therefore an upper bound on how clean evaluation can be.
- **Compute scope.** We use mid-sized encoders (ResNet-50, DistilBERT). Large vision-language models (CLIP ViT-L, BLIP) are out of scope for the mandatory part.

## References

1. X. Wang, D. Kumar, N. Thome, M. Cord, F. Precioso. *Recipe recognition with large multimodal food dataset.* ICME Workshops, 2015.
2. D. Kiela, S. Bhooshan, H. Firooz, E. Perez, D. Testuggine. *Supervised Multimodal Bitransformers for Classifying Images and Text.* arXiv:1909.02950, 2019. Code: `facebookresearch/mmbt`.
3. L. Bossard, M. Guillaumin, L. Van Gool. *Food-101 – Mining Discriminative Components with Random Forests.* ECCV, 2014.
4. V. Sanh, L. Debut, J. Chaumond, T. Wolf. *DistilBERT, a distilled version of BERT.* arXiv:1910.01108, 2019.
5. K. He, X. Zhang, S. Ren, J. Sun. *Deep Residual Learning for Image Recognition.* CVPR, 2016.
