# CO3133 Assignment 3 - UPMC Food-101 Image-Title Classification

Multimodal Deep Learning - Group G-roundhog, Semester 261, HCMUT.
101-way recipe classification from a web image **and** its page title: an image-only baseline (ResNet-50), a
text-only baseline (DistilBERT), a late fusion and a concat fusion (M2), and a cross-modal transformer fusion (M3),
with support / conflict analysis between the two modalities.

**Current milestone: M2 - draft** (unimodal A, unimodal B and simple fusion trained with 3 seeds, quantitative results).

| Document | Link |
| --- | --- |
| Dataset proposal (M1, approved) | [PROPOSAL.md](PROPOSAL.md) |
| M2 draft report (PDF, LaTeX source in `docs/a3-report/`) | [docs/a3-report/main.pdf](docs/a3-report/main.pdf) |
| M2 draft report (Markdown) | [docs/REPORT_M2.md](docs/REPORT_M2.md) |
| Assignment page | <https://pbt245.github.io/CO3133-DLIA/assignments/assignment3.html> |
| Results tables | [outputs/comparison/summary.md](outputs/comparison/summary.md) |
| EDA | [outputs/eda/EDA.md](outputs/eda/EDA.md) |
| Final report / slides / video | `[TODO: M3]` |

## Results (M2, test complete pairs, mean ± std over seeds 42 / 123 / 2026)

| Model | Accuracy | Macro-F1 | Macro-F1, title unseen in train |
| --- | ---: | ---: | ---: |
| K0 keyword rule (no training) | 76.95 | 83.56 | 82.89 |
| UA image - ResNet-50 | 70.73 ± 0.09 | 70.80 ± 0.06 | 70.59 ± 0.12 |
| UB text - DistilBERT | 86.42 ± 0.91 | 86.80 ± 0.25 | 82.57 ± 0.44 |
| F0 late fusion (no training) | **93.93 ± 0.23** | **93.86 ± 0.24** | **92.03 ± 0.32** |
| F1 concat fusion | 92.85 ± 0.06 | 92.80 ± 0.06 | 90.28 ± 0.03 |

Both fusions beat the best unimodal model under the decision rule. F1 loses to F0 because its image branch is under-used (report section 3.2).

## Repository structure

```
PROPOSAL.md                     M1 dataset proposal (approved)
docs/REPORT_M2.md               M2 draft report
configs/common.yaml             shared settings (data, text, train, optim, scheduler, eval)
configs/upmc/{ua_resnet50,ub_distilbert,f1_concat}.yaml   one file per model
src/data/text.py                title cleaning, validity rule, class-name lexicon, keyword rule, site names
src/data/upmc.py                titles, split files, multimodal Dataset / DataLoaders
src/data/phash.py               perceptual hash + Hamming search (from A2)
src/models/                     encoders (ResNet-50, DistilBERT), UA / UB / F1 classifiers, registry
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
scripts/compare.py              mean ± std, decision rule, leakage views, support / conflict
scripts/pack_results.py         zip results / checkpoints
scripts/smoke_test.py           end-to-end test on a tiny synthetic mirror (CPU, ~2 min)
scripts/make_kaggle_bundle.py   Kaggle datasets for Internet-off notebooks (code, weights)
scripts/remote/                 runner for a shared lab GPU (setup, tmux launch, status, upload)
data/splits/                    committed split: {train,val,test}.csv + split.json (fingerprints)
data/meta/                      per-item image / text metadata (no titles, no pixels)
data/audit/                     pair-audit verdicts
outputs/eda, outputs/comparison, outputs/runs/upmc/<model>/seed<s>/   committed results (no checkpoints)
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

- **Resume:** an interrupted run continues from `checkpoints/last.pt` with `--resume`; `run_experiments.py` does this automatically.
- **Re-evaluate** a finished run: `python -c "from src.engine.evaluate import evaluate_run; import torch; evaluate_run('outputs/runs/upmc/f1_concat/seed42', torch.device('cuda'))"`.

**Shared lab GPU (how the M2 runs were made).** On an Ubuntu machine reached through AnyDesk, with the repo cloned to `~/<work>/co3133-a3`:

```bash
bash scripts/remote/setup.sh          # venv on top of the machine's CUDA torch, Kaggle downloads, checks
bash scripts/remote/launch.sh         # whole M2 pipeline in tmux (picks the freest GPU, retries + resumes)
bash scripts/remote/status.sh         # progress
bash scripts/remote/upload_results.sh # results + checkpoints -> private Kaggle datasets
```

## 3. Configurations and seeds

- All hyper-parameters are in [configs/common.yaml](configs/common.yaml) and the three model files. Every run stores its resolved `config.yaml`, `env.json` (hardware, versions, git commit) and `train.log`.
- **Split seed 42**, fixed for every model (StratifiedGroupKFold over image-duplicate clusters). **Training seeds 42, 123, 2026.**
- Selection: best validation macro-F1 on complete pairs, early stopping patience 3.

## 4. Hardware and runtimes (M2)

| Step | Hardware | Time |
| --- | --- | --- |
| Data preparation + EDA + audit sheet | Kaggle CPU notebook, 4 vCPU | 26 min |
| 9 training runs + evaluation | NVIDIA RTX A6000 48 GB (shared lab machine), 64 cores, 16 loader workers; Python 3.13.13, torch 2.12.0+cu130, torchvision 0.27.0, transformers 5.23.0, CUDA 13.0 | 2.0 h (UA 14.4 min, UB 2.0 min, F1 18.4 min per run) |

## 5. Checkpoints and traceability

- `best.pt` of every run is in the private Kaggle dataset `bathanhphan/co3133-a3-m2-checkpoints` (2.1 GB, sha256 list in `CHECKPOINTS_SHA256.txt`). It will be made public with M3. Unzip at the repo root.
- Every reported number traces back to:
  - a config: `outputs/runs/upmc/<model>/seed<s>/config.yaml`
  - the split: fingerprints in `train_summary.json`, equal to `data/splits/split.json`
  - the checkpoint: sha256 in `eval/test_metrics.json`
  - the commit: `env.json`; M2 runs used commit `cfc509a`

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
| Checkpoints | private Kaggle dataset (public with M3) |
| Links to report and Pages | done (M2 draft; final report in M3) |
| Traceability | done (section 5) |

## Group

| Name | Student ID | GitHub |
| --- | --- | --- |
| Phan Bá Thanh | 2353084 | [pbt245](https://github.com/pbt245) |
| Trần Công Hoàng Phước | 2352966 | [BenjaminPhuoc](https://github.com/BenjaminPhuoc) |

Course: Deep Learning and Its Applications (CO3133), Semester 261, Ho Chi Minh City University of Technology (VNU-HCM),
Faculty of Computer Science and Engineering. Instructor: Lê Thành Sách.
