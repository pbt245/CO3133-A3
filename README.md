# CO3133 Assignment 3 - UPMC Food-101 Image-Text Classification

Multimodal Deep Learning - Group G-roundhog, Semester 261, HCMUT.
101-way recipe classification from a web image **and** its page title: an image-only baseline (ResNet-50),
a text-only baseline (DistilBERT), a simple concat fusion and an improved cross-modal transformer fusion,
with support/conflict analysis between the two modalities.

**Current milestone: M1 - Dataset proposal** (submitted 2026-11-14, approval pending).
No model code exists yet; this README will grow into the full reproducibility guide in M2/M3.

- Dataset proposal (M1): [PROPOSAL.md](PROPOSAL.md)
- Assignment page: <https://pbt245.github.io/CO3133-DLIA/assignments/assignment3.html>
- Report / slides / video / checkpoints: `[TODO: M3]`

| Item | Choice |
| --- | --- |
| Task option | Image-text classification (spec 6.2) |
| Dataset | UPMC Food-101, Kaggle mirror `gianmarco96/upmcfood101`: 90,704 images + page titles, 101 classes |
| Modalities | A = RGB image, B = web page title (same page, linked by file name `<label>_<num>`) |
| Split | official train → train / 5,000 val, stratified by class and grouped by image-duplicate cluster (seed 42); official test untouched + deduplicated test subset |
| Metrics | macro-F1 (primary), accuracy, top-5, ECE, support/conflict breakdown, compute cost |
| Models | K0 keyword rule, UA ResNet-50 (image only), UB DistilBERT (text only), F0 late fusion, F1 concat fusion, F2 cross-modal transformer fusion |
| Ablations | E1 auxiliary unimodal losses on/off, E2 class-name masking in titles, E-mod test-time modality removal (+ optional E3, E4) |

## Verified dataset facts (inventory run, 2026-11-14)

Measured with [scripts/inventory_upmc.py](scripts/inventory_upmc.py) on Kaggle (notebook `bathanhphan/co3133-a3-inventory`):

- **Images:** 67,988 train / 22,716 test, all RGB JPEG, 0 decode failures, 7.70 GB.
- **Titles:** 67,972 train / 22,716 test rows in `texts/{train,test}_titles.csv` (no header: file, title, label).
  - Every row matches exactly one image with the same label. 16 train images have no title.
  - The mirror contains **titles only**, not full recipe text: median 7 words.
- **Label leakage in text:** 73 % of titles contain their own class name. A no-learning keyword rule reaches 71 % test accuracy.
- **Duplicates:** 1,253 test images (5.5 %) are byte-identical to a train image. 437 duplicate groups carry two or more labels.

## Repository structure

```
PROPOSAL.md                  M1 dataset proposal (modalities, pair validity, split unit, metrics, models, compute)
scripts/
  inventory_upmc.py          read-only inventory of the Kaggle mirror (counts, CSV layout, matching, duplicates, leakage)
  make_kaggle_bundle.py      packs the repo into the private Kaggle dataset <user>/co3133-a3-code (Internet-off runs)
```

Planned for M2: `src/` (data, models, engine), `configs/`, `data/splits/`, EDA / pair-building / training scripts.

## Reproducing the inventory (M1)

All Kaggle runs use **Internet off**. Code reaches Kaggle as a private dataset, never via `git clone` or `pip`.

```bash
# 1. local machine: build and upload the code bundle (commit first, BUNDLE_INFO records the commit)
python3 scripts/make_kaggle_bundle.py
kaggle datasets create  -p outputs/kaggle_bundle/code                    # first time
kaggle datasets version -p outputs/kaggle_bundle/code -m "what changed"  # later updates
```

2. **Kaggle notebook `co3133-a3-inventory`:**
   - Settings: Accelerator *None*, Internet *Off*.
   - Inputs: `gianmarco96/upmcfood101` and `bathanhphan/co3133-a3-code`.
   - Cells:

```python
# setup (same cell in every A3 notebook): copy the code bundle to /kaggle/working/co3133-a3
import glob, os, shutil, zipfile
REPO = "/kaggle/working/co3133-a3"
hit = glob.glob("/kaggle/input/**/co3133-a3/BUNDLE_INFO", recursive=True)
if hit:
    shutil.copytree(os.path.dirname(hit[0]), REPO, dirs_exist_ok=True)
else:
    zipfile.ZipFile(glob.glob("/kaggle/input/**/co3133-a3-code.zip", recursive=True)[0]).extractall("/kaggle/working")
os.chdir(REPO)
print(open("BUNDLE_INFO").read())
```

```python
!python scripts/inventory_upmc.py --out /kaggle/working/inventory
```

3. *Save Version → Save & Run All*. Runtime is about 26 min on the 4-vCPU CPU notebook.
4. Fetch the results locally:

```bash
kaggle kernels output bathanhphan/co3133-a3-inventory -p outputs/inventory
```

The outputs are `summary.json`, `image_index.csv.gz` (per-image size, md5, decode status) and `text_index.csv.gz` (per-title length and matching flags, no text).

## Environment

- Local: Python ≥ 3.10 with `numpy`, `pandas`, `Pillow` (inventory only). No training on the laptop.
- Kaggle CPU notebook used for the inventory: Python 3.13.18, pandas 2.3.3, numpy 2.1.3, Pillow 13.0.0, 4 vCPU.
- Training (M2/M3): Kaggle NVIDIA RTX PRO 6000 Blackwell (95 GB), Internet off. Pretrained weights (ResNet-50, DistilBERT) are shipped inside the code bundle.

## Reproducibility checklist (status)

| Item | Status |
| --- | --- |
| README with install steps | M1 (this file) - full version in M2 |
| Dataset preparation scripts | inventory done; pair building, pre-resize and tokenisation in M2 |
| Training / evaluation commands, configs, seeds | M2 (seeds fixed in the proposal: split 42; runs 42, 123, 2026) |
| `requirements.txt` with versions | M2 |
| Hardware for reported runtimes | documented above; per run in `env.json` from M2 |
| Checkpoints | M3 |
| Links to report and GitHub Pages | Pages linked above; report in M3 |

## Group

| Name | Student ID | GitHub |
| --- | --- | --- |
| Phan Bá Thanh | 2353084 | [pbt245](https://github.com/pbt245) |
| Trần Công Hoàng Phước | 2352966 | [BenjaminPhuoc](https://github.com/BenjaminPhuoc) |

Course: Deep Learning and Its Applications (CO3133), Semester 261, Ho Chi Minh City University of Technology (VNU-HCM),
Faculty of Computer Science and Engineering. Instructor: Lê Thành Sách.
