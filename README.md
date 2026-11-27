# CO3133 Assignment 3 - UPMC Food-101 Image-Title Classification

Multimodal Deep Learning - Group G-roundhog, Semester 261, HCMUT.
101-way recipe classification from a web image **and** its page title: an image-only baseline (ResNet-50), a
text-only baseline (DistilBERT), a late fusion and a concat fusion (M2), and a cross-modal transformer fusion (M3),
with support / conflict analysis between the two modalities.

**Current milestone: M3 - final** (improved fusion F2, ablations E1 / E2 / E4, modality-removal curves, error taxonomy; every trained model with 3 seeds).

| Document | Link |
| --- | --- |
| Dataset proposal (M1, approved) | [PROPOSAL.md](PROPOSAL.md) |
| **Final report (PDF, LaTeX source in `docs/a3-report/`)** | [docs/a3-report/main.pdf](docs/a3-report/main.pdf) |
| Final report (Markdown) | [docs/REPORT_M3.md](docs/REPORT_M3.md) |
| M2 draft report (Markdown; data, audit, EDA in full) | [docs/REPORT_M2.md](docs/REPORT_M2.md) |
| Slides | [docs/a3-slides/slides.pdf](docs/a3-slides/slides.pdf) |
| Video | `[video URL]` |
| Assignment page | <https://pbt245.github.io/CO3133-DLIA/assignments/assignment3.html> |
| Results tables | [outputs/comparison/summary.md](outputs/comparison/summary.md) |
| M3 analyses (ablations, removal curves, support / conflict, calibration, attention) | [outputs/analysis/ANALYSIS.md](outputs/analysis/ANALYSIS.md) |
| EDA | [outputs/eda/EDA.md](outputs/eda/EDA.md) |

## Results (test complete pairs, mean ± std over seeds 42 / 123 / 2026)

| Model | Accuracy | Macro-F1 | Macro-F1, title unseen in train | Macro-F1, class names masked (E2) |
| --- | ---: | ---: | ---: | ---: |
| K0 keyword rule (no training) | 76.95 | 83.56 | 82.89 | - |
| T0 TF-IDF + logistic regression | 87.12 | 87.26 | 82.57 | 47.79 |
| UA image - ResNet-50 | 70.73 ± 0.09 | 70.80 ± 0.06 | 70.59 ± 0.12 | (70.80) |
| UB text - DistilBERT | 86.42 ± 0.91 | 86.80 ± 0.25 | 82.57 ± 0.44 | 43.70 ± 0.16 |
| F0 late fusion (no training) | **93.93 ± 0.23** | **93.86 ± 0.24** | **92.03 ± 0.32** | 78.85 ± 0.17 |
| F1 concat fusion | 92.85 ± 0.06 | 92.80 ± 0.06 | 90.28 ± 0.03 | 80.43 ± 0.14 |
| **F2 cross-modal fusion** | 93.58 ± 0.07 | 93.53 ± 0.07 | 91.05 ± 0.14 | **81.31 ± 0.02** |
| E1: F2 without auxiliary losses | 93.13 ± 0.08 | 93.07 ± 0.08 | 90.52 ± 0.12 | - |
| E4: F2 without modality dropout | 93.50 ± 0.06 | 93.45 ± 0.06 | 90.98 ± 0.09 | - |

- **Every fusion beats every unimodal model by 6-7 pp.**
- **F2 vs F1:** F2 beats F1 (+0.73 pp, McNemar p = 2e-9).
- **F2 vs F0:** F2 ties the late fusion F0 on plain titles (-0.33 pp, no decision). It beats F0 when the class names are masked (+2.46 pp, p = 2e-28).
- **E1:** the auxiliary losses help (+0.46 pp) and keep the image encoder as strong as a standalone image model.
- **E4:** the modality dropout costs nothing on complete pairs and adds +4.7 pp when the title is missing.

The full analysis is in the [final report](docs/a3-report/main.pdf).

## Repository structure

```
PROPOSAL.md                     M1 dataset proposal (approved)
docs/REPORT_M2.md               M2 draft report (data, audit, EDA, split in full)
docs/REPORT_M3.md               final report (Markdown); docs/a3-report/ = LaTeX source + PDF; docs/a3-slides/ = slides
docs/video_script.md            script of the presentation video
configs/common.yaml             shared settings (data, text, train, optim, scheduler, eval)
configs/upmc/{ua_resnet50,ub_distilbert,f1_concat,f2_crossmodal}.yaml   one file per model
configs/upmc/ablations/         E1 (f2_noaux), E4 (f2_nomoddrop), E2 (*_mask: class names masked)
src/data/text.py                title cleaning, validity rule, class-name lexicon, keyword rule, site names
src/data/upmc.py                titles, split files, multimodal Dataset / DataLoaders
src/data/phash.py               perceptual hash + Hamming search (from A2)
src/models/                     encoders (ResNet-50, DistilBERT), UA / UB / F1 / F2 classifiers, registry
src/engine/                     train / predict loops, trainer, evaluation, metrics, report helpers, inference cost
src/utils/                      config, seeds, paths (offline inputs), environment / git info, logging
scripts/inventory_upmc.py       M1 read-only inventory of the Kaggle mirror
scripts/prepare_data.py         pairs, validity, duplicate clusters, split, pre-resized images
scripts/eda.py                  EDA tables and figures
scripts/audit_pairs.py          200-pair manual audit (HTML labelling page + scoring)
scripts/train.py                train one config and seed, then evaluate on test
scripts/run_experiments.py      many configs x seeds; skips finished runs, resumes unfinished ones
scripts/keyword_baseline.py     K0
scripts/late_fusion.py          F0 (weight tuned on val)
scripts/tfidf_baseline.py       T0 (and T0 on masked titles, --mask)
scripts/removal_predictions.py  predictions with one modality removed, for runs evaluated before M3
scripts/qualitative.py          [FUS] attention over the test set + Grad-CAM example panels
scripts/conflict_review.py      100-item conflict review (HTML labelling page + scoring)
scripts/analysis_m3.py          M3 analyses: support / conflict over seeds, removal curves, ablation tables, calibration
scripts/compare.py              mean ± std, decision rule, leakage views, support / conflict
scripts/pack_results.py         zip results / checkpoints
scripts/smoke_test.py           end-to-end test on a tiny synthetic mirror (CPU, ~2 min)
scripts/make_kaggle_bundle.py   Kaggle datasets for Internet-off notebooks (code, weights)
scripts/remote/                 runner for a shared lab GPU (setup, tmux launch, status, upload)
data/splits/                    committed split: {train,val,test}.csv + split.json (fingerprints)
data/meta/                      per-item image / text metadata (no titles, no pixels)
data/audit/                     pair-audit and conflict-review verdicts
outputs/eda, outputs/comparison, outputs/analysis, outputs/runs/upmc/<model>/seed<s>/   committed results (no checkpoints)
```

## Installation

Python >= 3.10. With an NVIDIA GPU, install the CUDA build of `torch` / `torchvision` first, then:

```bash
pip install -r requirements.txt
```

Pretrained weights are loaded from local files (no download at run time):

```bash
python scripts/make_kaggle_bundle.py weights   # downloads once into ./weights/ (ResNet-50 IMAGENET1K_V2, distilbert-base-uncased)
```

Inputs are found automatically (`src/utils/paths.py`). They can also be set explicitly:

| Variable | Meaning |
| --- | --- |
| `UPMC_ROOT` | mirror folder with `texts/{train,test}_titles.csv` (and `images/` for data preparation) |
| `UPMC_IMAGE_ROOT` | pre-resized image folder `upmc256/` (`images/{train,test}/<label>/<id>.jpg`) |
| `A3_WEIGHTS_DIR` | folder with `resnet50-11ad3fa6.pth` and `distilbert-base-uncased/` |

Check the code without any real data (CPU, about 2 minutes): `python scripts/smoke_test.py --workdir /tmp/a3_smoke`.

## 1. Data preparation (done once; outputs are committed)

Kaggle CPU notebook **`bathanhphan/co3133-a3-prepare`** (Internet off). Inputs: `gianmarco96/upmcfood101`, `bathanhphan/co3133-a3-code`, `bathanhphan/co3133-a3-weights`. Total time 26 min on 4 vCPU.

```python
# setup cell (every A3 Kaggle notebook)
import glob, os, shutil, zipfile
REPO = "/kaggle/working/co3133-a3"
hit = [h for h in glob.glob("/kaggle/input/**/co3133-a3/BUNDLE_INFO", recursive=True) if "co3133-a3-code" in h]
if hit:
    shutil.copytree(os.path.dirname(hit[0]), REPO, dirs_exist_ok=True)
else:
    zipfile.ZipFile(glob.glob("/kaggle/input/**/co3133-a3-code.zip", recursive=True)[0]).extractall("/kaggle/working")
os.chdir(REPO)
print(open("BUNDLE_INFO").read())
```
```bash
python scripts/prepare_data.py --out /kaggle/working/prepare --resized /kaggle/working/upmc256.zip
python scripts/eda.py --split-dir /kaggle/working/prepare/data/splits --meta-dir /kaggle/working/prepare/data/meta --out /kaggle/working/prepare/eda
python scripts/audit_pairs.py make --split-dir /kaggle/working/prepare/data/splits --out /kaggle/working/prepare/audit
```

`prepare/data/{splits,meta}` are copied into the repo and committed. `upmc256.zip` (3.2 GB) is the image input for training. Pair audit: open `audit/index.html`, label, download the CSV into `data/audit/`, then `python scripts/audit_pairs.py score`.

The code bundle for Kaggle is built from a clean commit (`BUNDLE_INFO` records the commit):

```bash
python scripts/make_kaggle_bundle.py code && kaggle datasets version -p outputs/kaggle_bundle/code -m "message"
```

## 2. Training and evaluation

One run (trains, selects the best val macro-F1 epoch, evaluates it on test):

```bash
python scripts/train.py --config configs/upmc/ua_resnet50.yaml --seed 42
python scripts/train.py --config configs/upmc/ub_distilbert.yaml --seed 42
python scripts/train.py --config configs/upmc/f1_concat.yaml --seed 42 --set data.num_workers=16
```

All M2 runs, then the zero-training references and the comparison:

```bash
python scripts/run_experiments.py --models ua_resnet50 ub_distilbert f1_concat --seeds 42 123 2026 --set data.num_workers=16 train.show_progress=false
python scripts/keyword_baseline.py
python scripts/late_fusion.py --seeds 42 123 2026
python scripts/compare.py
```

All M3 runs and analyses (F2, ablations, T0, figures, review page):

```bash
python scripts/run_experiments.py --models f2_crossmodal ub_distilbert_mask f1_concat_mask f2_crossmodal_mask f2_noaux f2_nomoddrop --seeds 42 123 2026 --set data.num_workers=16 train.show_progress=false
python scripts/late_fusion.py --text ub_distilbert_mask --name f0_late_fusion_mask --seeds 42 123 2026
python scripts/tfidf_baseline.py && python scripts/tfidf_baseline.py --mask
python scripts/removal_predictions.py outputs/runs/upmc/f1_concat/seed{42,123,2026}
python scripts/qualitative.py --runs outputs/runs/upmc/{f2_crossmodal,f2_noaux,f2_nomoddrop,f2_crossmodal_mask}/seed42
python scripts/conflict_review.py make --out outputs/review      # then label, save data/audit/conflict_review.csv
python scripts/conflict_review.py score
python scripts/compare.py && python scripts/analysis_m3.py
```

- **Resume:** an interrupted run continues from `checkpoints/last.pt` with `--resume`; `run_experiments.py` does this automatically.
- **Re-evaluate** a finished run: `python -c "from src.engine.evaluate import evaluate_run; import torch; evaluate_run('outputs/runs/upmc/f1_concat/seed42', torch.device('cuda'))"`.

**Shared lab GPU (how the M2 and M3 runs were made).** On an Ubuntu machine reached through AnyDesk, with the repo cloned to `~/<work>/co3133-a3`:

```bash
bash scripts/remote/setup.sh          # venv on top of the machine's CUDA torch, Kaggle downloads, checks
bash scripts/remote/launch.sh m3      # whole pipeline in tmux (m2 or m3; picks the freest GPU, retries + resumes)
bash scripts/remote/status.sh         # progress
A3_MILESTONE=m3 bash scripts/remote/upload_results.sh   # results + checkpoints -> Kaggle datasets
```

## 3. Configurations and seeds

- All hyper-parameters are in [configs/common.yaml](configs/common.yaml), the model files and `configs/upmc/ablations/`. Every run stores its resolved `config.yaml`, `env.json` (hardware, versions, git commit) and `train.log`.
- **Split seed 42**, fixed for every model (StratifiedGroupKFold over image-duplicate clusters). **Training seeds 42, 123, 2026.**
- Selection: best validation macro-F1 on complete pairs, early stopping patience 3.

## 4. Hardware and runtimes

| Step | Hardware | Time |
| --- | --- | --- |
| Data preparation + EDA + audit sheet | Kaggle CPU notebook, 4 vCPU | 26 min |
| M2: 9 training runs + evaluation | NVIDIA RTX A6000 48 GB (shared lab machine), 64 cores, 16 loader workers; Python 3.13.13, torch 2.12.0+cu130, torchvision 0.27.0, transformers 5.23.0, CUDA 13.0 | 2.0 h (UA 14.4 min, UB 2.0 min, F1 18.4 min per run) |
| M3: 18 training runs + evaluation + analyses | same machine | about 5.7 h (F2-type 18.6 min, F1-mask 18.3 min, UB-mask 2.2 min per run; T0 4.4 min on CPU) |

## 5. Checkpoints and traceability

- `best.pt` of every run is on Kaggle, with a sha256 list (`*_SHA256.txt`). Unzip at the repo root.
  - M2 (UA, UB, F1): <https://www.kaggle.com/datasets/bathanhphan/co3133-a3-m2-checkpoints>
  - M3 (F2, E1, E4, E2 models): <https://www.kaggle.com/datasets/bathanhphan/co3133-a3-m3-checkpoints>
- Every reported number traces back to:
  - a config: `outputs/runs/upmc/<model>/seed<s>/config.yaml`
  - the split: fingerprints in `train_summary.json`, equal to `data/splits/split.json`
  - the checkpoint: sha256 in `eval/test_metrics.json`
  - the commit: `env.json`; M2 runs used commit `cfc509a`, M3 runs commit `b51d3fb` (clean working tree)

## Reproducibility checklist (spec 7)

| Item | Status |
| --- | --- |
| README with installation steps | done |
| Dataset preparation scripts | done (`prepare_data.py`, `eda.py`, `audit_pairs.py`) |
| Training / evaluation commands | done (sections 1-2) |
| Configuration files | done (`configs/`) |
| Random seeds | done (split 42; 42, 123, 2026) |
| Dependency versions | `requirements.txt` + exact versions per run in `env.json` |
| Hardware for runtimes | done (section 4) |
| Checkpoints | Kaggle datasets (section 5) |
| Links to report and Pages | done |
| Traceability | done (section 5) |

## Group

| Name | Student ID | GitHub |
| --- | --- | --- |
| Phan Bá Thanh | 2353084 | [pbt245](https://github.com/pbt245) |
| Trần Công Hoàng Phước | 2352966 | [BenjaminPhuoc](https://github.com/BenjaminPhuoc) |

Course: Deep Learning and Its Applications (CO3133), Semester 261, Ho Chi Minh City University of Technology (VNU-HCM),
Faculty of Computer Science and Engineering. Instructor: Lê Thành Sách.
