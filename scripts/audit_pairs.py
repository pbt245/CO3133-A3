#!/usr/bin/env python
"""Manual pair-validity audit (proposal 2.3, criterion 4): 200 train pairs = 10 pairs from each of 20 random classes.

  # Kaggle (needs the images): thumbnails + an HTML labelling page
  python scripts/audit_pairs.py make --split-dir /kaggle/working/prepare/data/splits --out /kaggle/working/prepare/audit
  # local: open audit/index.html, choose a verdict per pair, press "Download CSV",
  # save it as data/audit/pair_audit.csv, then
  python scripts/audit_pairs.py score

Verdicts: valid | image_unrelated (logo, ad, person, collage, other dish) | text_unrelated (boilerplate, another
recipe) | both_unrelated | wrong_label (the pair is consistent but the class label is wrong).
The committed CSV holds id, label, verdict, note: no titles, no pixels.
"""
import argparse
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from PIL import Image

from src.data.text import clean_title
from src.data.upmc import image_file, read_titles
from src.engine.metrics import wilson_ci
from src.utils import paths
from src.utils.io import resolve_path, save_json

VERDICTS = ["valid", "image_unrelated", "text_unrelated", "both_unrelated", "wrong_label"]

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>UPMC pair audit</title>
<style>body{{font-family:system-ui,sans-serif;margin:16px;background:#fafafa;color:#222}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px}}
.card{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:8px}}
.card img{{width:100%;height:220px;object-fit:contain;background:#eee}}
.t{{font-size:14px;margin:6px 0}} .l{{font-size:12px;color:#555}} select{{width:100%;margin-top:4px}}
input{{width:100%;margin-top:4px;box-sizing:border-box}} .bar{{position:sticky;top:0;background:#fafafa;padding:8px 0}}
.done{{border-color:#2a7}}</style></head><body>
<div class="bar"><b>UPMC Food-101 pair audit</b> - {n} pairs. Verdicts: valid / image unrelated / text unrelated /
both unrelated / wrong label. <button onclick="dl()">Download CSV</button> <span id="cnt"></span></div>
<div class="grid">{cards}</div>
<script>
const KEY='upmc_pair_audit_v1';
function cards(){{return document.querySelectorAll('.card');}}
function save(){{const s={{}};cards().forEach(c=>{{s[c.dataset.id]=[c.querySelector('select').value,c.querySelector('input').value];}});
try{{localStorage.setItem(KEY,JSON.stringify(s));}}catch(e){{}}}}
function load(){{try{{const s=JSON.parse(localStorage.getItem(KEY)||'{{}}');cards().forEach(c=>{{const v=s[c.dataset.id];
if(v){{c.querySelector('select').value=v[0];c.querySelector('input').value=v[1];}}}});}}catch(e){{}}}}
function upd(){{let n=0;cards().forEach(c=>{{const s=c.querySelector('select').value;c.classList.toggle('done',!!s);if(s)n++;}});
document.getElementById('cnt').textContent=n+' / '+cards().length+' labelled (auto-saved in this browser)';save();}}
document.querySelectorAll('select,input').forEach(e=>{{e.addEventListener('change',upd);e.addEventListener('input',upd);}});
load();upd();
function dl(){{let rows=['id,label,verdict,note'];cards().forEach(c=>{{
const note=c.querySelector('input').value.replaceAll('"','""');
rows.push([c.dataset.id,c.dataset.label,c.querySelector('select').value,'"'+note+'"'].join(','));}});
const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([rows.join('\\n')+'\\n'],{{type:'text/csv'}}));
a.download='pair_audit.csv';a.click();}}
</script></body></html>"""

CARD = """<div class="card" data-id="{id}" data-label="{label}"><img src="img/{id}.jpg" loading="lazy">
<div class="t">{k}. {title}</div><div class="l">label: <b>{label}</b></div>
<select><option value="">- verdict -</option>{opts}</select><input placeholder="note (optional)"></div>"""


def make(args):
    split = pd.read_csv(resolve_path(args.split_dir) / "train.csv", dtype={"id": str})
    root = paths.upmc_root(args.upmc_root)
    titles = read_titles(root).set_index("id")["title"]
    split = split[split["id"].isin(titles.index)]
    rng = np.random.default_rng(args.seed)
    classes = sorted(split["label"].unique())
    chosen = sorted(str(c) for c in rng.choice(classes, size=args.n_classes, replace=False))
    rows = []
    for c in chosen:
        sub = split[split["label"] == c]
        rows.append(sub.iloc[np.sort(rng.choice(len(sub), size=args.per_class, replace=False))])
    sample = pd.concat(rows, ignore_index=True)
    out = Path(args.out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    cards = []
    opts = "".join(f'<option value="{v}">{v.replace("_", " ")}</option>' for v in VERDICTS)
    for k, r in enumerate(sample.itertuples(), 1):
        with Image.open(image_file(root, r.official_split, r.label, r.id)) as im:
            im = im.convert("RGB")
            im.thumbnail((360, 360))
            im.save(out / "img" / f"{r.id}.jpg", quality=85)
        cards.append(CARD.format(id=r.id, label=r.label, k=k, opts=opts,
                                 title=html.escape(clean_title(titles[r.id])) or "<i>(empty)</i>"))
    (out / "index.html").write_text(PAGE.format(n=len(sample), cards="\n".join(cards), verdicts=json.dumps(VERDICTS)),
                                    encoding="utf-8")
    sample.assign(verdict="", note="")[["id", "label", "verdict", "note"]].to_csv(out / "pair_audit_template.csv",
                                                                                  index=False)
    print(f"{len(sample)} pairs from classes {chosen} -> {out}/index.html")


def score(args):
    df = pd.read_csv(resolve_path(args.csv), dtype=str, keep_default_na=False)
    bad = sorted(set(df["verdict"]) - set(VERDICTS))
    if bad:
        sys.exit(f"unknown or empty verdicts {bad}: label every pair first")
    n = len(df)
    rows = []
    for v in VERDICTS:
        k = int((df["verdict"] == v).sum())
        lo, hi = wilson_ci(k, n)
        rows.append({"verdict": v, "count": k, "rate": k / n, "wilson95_low": lo, "wilson95_high": hi})
    image_bad = int(df["verdict"].isin(["image_unrelated", "both_unrelated"]).sum())
    text_bad = int(df["verdict"].isin(["text_unrelated", "both_unrelated"]).sum())
    summary = {"n": n, "classes": sorted(df["label"].unique()), "verdicts": rows,
               "image_uninformative": {"count": image_bad, "rate": image_bad / n, "wilson95": wilson_ci(image_bad, n)},
               "text_uninformative": {"count": text_bad, "rate": text_bad / n, "wilson95": wilson_ci(text_bad, n)}}
    out = resolve_path(args.out)
    save_json(summary, out)
    print("| verdict | count | rate | Wilson 95 % CI |\n| --- | ---: | ---: | --- |")
    for r in rows:
        print(f"| {r['verdict']} | {r['count']} | {100 * r['rate']:.1f} % | "
              f"{100 * r['wilson95_low']:.1f}-{100 * r['wilson95_high']:.1f} % |")
    print(f"-> {out}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--split-dir", default="data/splits")
    m.add_argument("--upmc-root", default="auto")
    m.add_argument("--out", default="outputs/audit")
    m.add_argument("--n-classes", type=int, default=20)
    m.add_argument("--per-class", type=int, default=10)
    m.add_argument("--seed", type=int, default=42)
    s = sub.add_parser("score")
    s.add_argument("--csv", default="data/audit/pair_audit.csv")
    s.add_argument("--out", default="data/audit/pair_audit_summary.json")
    args = p.parse_args()
    make(args) if args.cmd == "make" else score(args)


if __name__ == "__main__":
    main()
