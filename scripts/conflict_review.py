#!/usr/bin/env python
"""Error taxonomy of conflict cases from a manual review (proposal 5.5 item 5): 100 complete test pairs where the
two unimodal models (seed 42) disagree about correctness, 50 Conflict-I (image model right, text model wrong) and
50 Conflict-T (text model right, image model wrong), drawn at random (seed 42) and shown in shuffled order.

  # training machine (needs the images): thumbnails + an HTML labelling page
  python scripts/conflict_review.py make --out outputs/review
  # laptop: open review/index.html, choose the main cause per item, press "Download CSV",
  # save it as data/audit/conflict_review.csv, then
  python scripts/conflict_review.py score

The reviewer labels WHY the failing modality's model is wrong (Conflict-I: the text model; Conflict-T: the image
model), with one main cause:
  image_unrelated    the image does not show the dish (logo, ad, person, collage, step photo, text banner)
  visually_similar   the image shows the dish but looks like the predicted class (similar dishes, atypical plating)
  title_generic      the title carries no dish information (site name only, "Recipes", a person's name)
  title_other_dish   the title names a derived dish or another class ("Apple Pie Egg Rolls")
  wrong_label        the item really belongs to another class (the label is wrong)
  other              none of the above (explain in the note)
The committed CSV holds id, group, verdict, note: no titles, no pixels.
"""
import argparse
import html
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from PIL import Image

from src.data.text import clean_title
from src.data.upmc import image_file, load_split, read_titles
from src.engine.metrics import wilson_ci
from src.utils import paths
from src.utils.io import resolve_path, save_json

VERDICTS = ["image_unrelated", "visually_similar", "title_generic", "title_other_dish", "wrong_label", "other"]
FAILING = {"conflict_I": "text model wrong (title misleading?)", "conflict_T": "image model wrong (image misleading?)"}

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>UPMC conflict review</title>
<style>body{{font-family:system-ui,sans-serif;margin:16px;background:#fafafa;color:#222}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:12px}}
.card{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:8px}}
.card img{{width:100%;height:220px;object-fit:contain;background:#eee}}
.t{{font-size:14px;margin:6px 0}} .l{{font-size:12px;color:#444;line-height:1.5}} select{{width:100%;margin-top:4px}}
input{{width:100%;margin-top:4px;box-sizing:border-box}} .bar{{position:sticky;top:0;background:#fafafa;padding:8px 0;z-index:2}}
.g{{font-size:12px;font-weight:600;padding:2px 6px;border-radius:4px;display:inline-block;margin-bottom:4px}}
.conflict_I{{background:#fde2c8}} .conflict_T{{background:#cfe3fb}} .done{{border-color:#2a7}}
.ok{{color:#1a7f37}} .bad{{color:#c62828}}</style></head><body>
<div class="bar"><b>UPMC Food-101 conflict review</b> - {n} test items. For each item choose the MAIN reason why the
<u>failing</u> model is wrong (orange = text model wrong, blue = image model wrong).
Items 1-50: Thanh, 51-100: Phước. <button onclick="dl()">Download CSV</button> <span id="cnt"></span>
<details><summary>verdicts</summary><ul>
<li><b>image unrelated</b>: the image does not show the dish (logo, ad, person, collage, step photo, banner)</li>
<li><b>visually similar</b>: the image shows the dish but looks like the predicted class</li>
<li><b>title generic</b>: the title has no dish information (site name only, "Recipes")</li>
<li><b>title other dish</b>: the title names a derived dish or another class</li>
<li><b>wrong label</b>: the item really belongs to another class</li>
<li><b>other</b>: write a short note</li></ul></details></div>
<div class="grid">{cards}</div>
<script>
const KEY='upmc_conflict_review_v1';
function cards(){{return document.querySelectorAll('.card');}}
function save(){{const s={{}};cards().forEach(c=>{{s[c.dataset.id]=[c.querySelector('select').value,c.querySelector('input').value];}});
try{{localStorage.setItem(KEY,JSON.stringify(s));}}catch(e){{}}}}
function load(){{try{{const s=JSON.parse(localStorage.getItem(KEY)||'{{}}');cards().forEach(c=>{{const v=s[c.dataset.id];
if(v){{c.querySelector('select').value=v[0];c.querySelector('input').value=v[1];}}}});}}catch(e){{}}}}
function upd(){{let n=0;cards().forEach(c=>{{const s=c.querySelector('select').value;c.classList.toggle('done',!!s);if(s)n++;}});
document.getElementById('cnt').textContent=n+' / '+cards().length+' labelled (auto-saved in this browser)';save();}}
document.querySelectorAll('select,input').forEach(e=>{{e.addEventListener('change',upd);e.addEventListener('input',upd);}});
load();upd();
function dl(){{let rows=['id,group,verdict,note'];cards().forEach(c=>{{
const note=c.querySelector('input').value.replaceAll('"','""');
rows.push([c.dataset.id,c.dataset.group,c.querySelector('select').value,'"'+note+'"'].join(','));}});
const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([rows.join('\\n')+'\\n'],{{type:'text/csv'}}));
a.download='conflict_review.csv';a.click();}}
</script></body></html>"""

CARD = """<div class="card" data-id="{id}" data-group="{group}"><img src="img/{id}.jpg" loading="lazy">
<span class="g {group}">{k}. {group_txt}</span>
<div class="t">"{title}"</div><div class="l">label: <b>{label}</b><br>
image model (UA): <span class="{ua_c}">{ua}</span><br>text model (UB): <span class="{ub_c}">{ub}</span>{f2}</div>
<select><option value="">- main cause -</option>{opts}</select><input placeholder="note (optional)"></div>"""


def preds(run_dir):
    return pd.read_csv(Path(run_dir) / "eval" / "predictions_test.csv", dtype={"id": str}).set_index("id")


def make(args):
    _, frames = load_split(resolve_path(args.split_dir))
    test = frames["test"].set_index("id")
    ua, ub = preds(resolve_path(args.image_run)), preds(resolve_path(args.text_run))
    f2 = preds(resolve_path(args.fusion_run)) if (resolve_path(args.fusion_run) / "eval/predictions_test.csv").exists() else None
    ids = ua.index
    a_ok, b_ok = ua["correct"].astype(bool), ub.loc[ids, "correct"].astype(bool)
    rng = np.random.default_rng(args.seed)
    picks = []
    for gname, m in (("conflict_I", a_ok & ~b_ok), ("conflict_T", ~a_ok & b_ok)):
        pool = np.asarray(ids[m.to_numpy()])
        picks += [(i, gname) for i in rng.choice(pool, size=min(args.per_group, len(pool)), replace=False)]
    order = rng.permutation(len(picks))
    sample = pd.DataFrame([picks[i] for i in order], columns=["id", "group"])
    root = paths.upmc_root(args.upmc_root)
    img_root = paths.image_root(args.image_root)      # pre-resized upmc256 copy if present (as in training)
    titles = read_titles(root).set_index("id")["title"]
    out = Path(args.out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    opts = "".join(f'<option value="{v}">{v.replace("_", " ")}</option>' for v in VERDICTS)
    cards = []
    for k, r in enumerate(sample.itertuples(), 1):
        t = test.loc[r.id]
        with Image.open(image_file(img_root, t.official_split, t.label, r.id)) as im:
            im = im.convert("RGB")
            im.thumbnail((360, 360))
            im.save(out / "img" / f"{r.id}.jpg", quality=85)
        f2_txt = ""
        if f2 is not None:
            ok = bool(f2.loc[r.id, "correct"])
            f2_txt = f'<br>fusion F2: <span class="{"ok" if ok else "bad"}">{f2.loc[r.id, "pred_name"]}</span>'
        cards.append(CARD.format(id=r.id, group=r.group, k=k, group_txt=FAILING[r.group], label=t.label,
                                 title=html.escape(clean_title(titles[r.id])) or "<i>(empty)</i>",
                                 ua=ua.loc[r.id, "pred_name"], ua_c="ok" if a_ok[r.id] else "bad",
                                 ub=ub.loc[r.id, "pred_name"], ub_c="ok" if b_ok[r.id] else "bad", f2=f2_txt, opts=opts))
    (out / "index.html").write_text(PAGE.format(n=len(sample), cards="\n".join(cards)), encoding="utf-8")
    sample.assign(verdict="", note="").to_csv(out / "conflict_review_template.csv", index=False)
    print(f"{len(sample)} conflict items ({args.per_group} per group) -> {out}/index.html")


def score(args):
    df = pd.read_csv(resolve_path(args.csv), dtype=str, keep_default_na=False)
    bad = sorted(set(df["verdict"]) - set(VERDICTS))
    if bad:
        sys.exit(f"unknown or empty verdicts {bad}: label every item first")
    summary = {"n": len(df), "groups": {}}
    print("| group | verdict | count | rate | Wilson 95 % CI |\n| --- | --- | ---: | ---: | --- |")
    for gname, sub in [("all", df), *df.groupby("group")]:
        n = len(sub)
        rows = []
        for v in VERDICTS:
            k = int((sub["verdict"] == v).sum())
            lo, hi = wilson_ci(k, n)
            rows.append({"verdict": v, "count": k, "rate": k / n, "wilson95_low": lo, "wilson95_high": hi})
            if k:
                print(f"| {gname} | {v} | {k} | {100 * k / n:.0f} % | {100 * lo:.0f}-{100 * hi:.0f} % |")
        summary["groups"][gname] = {"n": n, "verdicts": rows}
    by_verdict = verdict_vs_models(df, args)
    if by_verdict is not None:
        summary["by_verdict"] = by_verdict.reset_index().to_dict(orient="records")
        t = by_verdict.reset_index()
        print("\n| " + " | ".join(t.columns) + " |\n|" + " --- |" * len(t.columns))
        for r in t.itertuples(index=False):
            print("| " + " | ".join(f"{v:.1f}" if isinstance(v, float) else str(v) for v in r) + " |")
    save_json(summary, resolve_path(args.out))
    print(f"-> {args.out}")


def verdict_vs_models(df, args):
    """Per (group, verdict): how often the fusion models are right, how confident the model that is right is
    (UA for Conflict-I, UB for Conflict-T), and the share of titles that occur verbatim in train/val."""
    runs = {"F0": args.f0_run, "F1": args.f1_run, "F2": args.fusion_run}
    runs = {k: resolve_path(v) for k, v in runs.items() if (resolve_path(v) / "eval/predictions_test.csv").exists()}
    ua, ub = resolve_path(args.image_run), resolve_path(args.text_run)
    if not runs or not (ub / "eval/predictions_test.csv").exists():
        return None
    d = df.set_index("id")
    if not d.index.isin(preds(ub).index).all():
        print("note: review items not in the given runs' predictions; per-verdict table skipped")
        return None
    for k, r in runs.items():
        d[f"{k} right"] = preds(r).loc[d.index, "correct"].astype(bool) * 100.0
    right_conf = np.where(d["group"] == "conflict_I", preds(ua).loc[d.index, "confidence"],
                          preds(ub).loc[d.index, "confidence"])
    d["right model conf"] = right_conf * 100.0
    tm = pd.read_csv(resolve_path(args.text_meta), dtype=str, keep_default_na=False)
    trainval = set(pd.concat([pd.read_csv(resolve_path(args.split_dir) / f"{s}.csv", dtype=str)
                              for s in ("train", "val")])["id"])
    seen = set(tm.loc[tm["id"].isin(trainval) & (tm["norm_md5"] != ""), "norm_md5"])
    md5 = tm.set_index("id")["norm_md5"]
    d["title seen %"] = d.index.map(lambda i: md5.get(i, "") in seen and md5.get(i, "") != "") * 100.0
    cols = [*(f"{k} right" for k in runs), "right model conf", "title seen %"]
    g = d.groupby(["group", "verdict"])
    out = g[cols].mean()
    out.insert(0, "n", g.size())
    return out


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--split-dir", default="data/splits")
    m.add_argument("--upmc-root", default="auto", help="mirror with texts/ (titles)")
    m.add_argument("--image-root", default="auto", help="folder with images/ (default: upmc256 copy, else mirror)")
    m.add_argument("--image-run", default="outputs/runs/upmc/ua_resnet50/seed42")
    m.add_argument("--text-run", default="outputs/runs/upmc/ub_distilbert/seed42")
    m.add_argument("--fusion-run", default="outputs/runs/upmc/f2_crossmodal/seed42")
    m.add_argument("--out", default="outputs/review")
    m.add_argument("--per-group", type=int, default=50)
    m.add_argument("--seed", type=int, default=42)
    s = sub.add_parser("score")
    s.add_argument("--csv", default="data/audit/conflict_review.csv")
    s.add_argument("--out", default="data/audit/conflict_review_summary.json")
    s.add_argument("--split-dir", default="data/splits")
    s.add_argument("--text-meta", default="data/meta/text_meta.csv.gz")
    s.add_argument("--image-run", default="outputs/runs/upmc/ua_resnet50/seed42")
    s.add_argument("--text-run", default="outputs/runs/upmc/ub_distilbert/seed42")
    s.add_argument("--f0-run", default="outputs/runs/upmc/f0_late_fusion/seed42")
    s.add_argument("--f1-run", default="outputs/runs/upmc/f1_concat/seed42")
    s.add_argument("--fusion-run", default="outputs/runs/upmc/f2_crossmodal/seed42")
    args = p.parse_args()
    make(args) if args.cmd == "make" else score(args)


if __name__ == "__main__":
    main()
