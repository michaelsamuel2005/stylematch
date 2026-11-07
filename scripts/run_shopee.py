"""Matching on REAL listings: Kaggle Shopee Price Match Guarantee (non-commercial use; not in the repo).
  PYTHONPATH=src python scripts/run_shopee.py                 # full run: TF-IDF, MiniLM, CLIP image, fusion
  PYTHONPATH=src python scripts/run_shopee.py --limit 3000    # quick check on a subsample of whole label groups
  PYTHONPATH=src python scripts/run_shopee.py --no-clip       # text only (no torch needed)
Embeddings are cached in data/cache/ so a rerun is instant."""
import argparse, json, os, time, numpy as np
from stylematch import real as R
from stylematch.match import TfidfEncoder, _norm

ap = argparse.ArgumentParser(); ap.add_argument("--raw", default="data/raw/shopee"); ap.add_argument("--limit", type=int)
ap.add_argument("--no-clip", action="store_true"); ap.add_argument("--no-minilm", action="store_true"); ap.add_argument("--k", type=int, default=10)
a = ap.parse_args()
df = R.shopee_frame(a.raw, a.limit); tag = f"n{len(df)}"; g = df.label_group.values
print(f"{len(df):,} postings in {df.label_group.nunique():,} groups; {(df.groupby('label_group').size() > 1).sum():,} groups have a match")
res = {"dataset": "Shopee Price Match Guarantee (Kaggle), train split", "postings": int(len(df)), "groups": int(df.label_group.nunique()),
       "k": a.k, "train_csv_sha": R.file_sha(os.path.join(a.raw, "train.csv")), "timings_s": {}}
E = {}
t = time.time(); E["text_tfidf_char"] = TfidfEncoder(256).fit_transform(df.title); res["timings_s"]["tfidf"] = round(time.time() - t, 1)
if not a.no_minilm:
    try:
        from stylematch.match import SentenceTransformerEncoder
        t = time.time(); E["text_minilm"] = R.cached(f"data/cache/shopee_minilm_{tag}.npy", lambda: SentenceTransformerEncoder().fit_transform(df.title))
        res["timings_s"]["minilm"] = round(time.time() - t, 1)
    except Exception as e: print("MiniLM skipped:", e)
if not a.no_clip:
    enc, name = R.image_encoder(); slug = name.replace(":", "_")
    t = time.time(); E["image_clip"] = R.cached(f"data/cache/shopee_{slug}_{tag}.npy", lambda: enc(list(df.path), log=print))
    res["timings_s"]["clip"] = round(time.time() - t, 1); res["clip_model"] = name
res["rows"] = R.evaluate_rows(E, g, a.k)
for tname in [n for n in ("text_minilm", "text_tfidf_char") if n in E]:
    if "image_clip" in E:
        res["rows"].update({f"{tname}+image_clip_w{w}": v for k, v in R.fusion_rows(E[tname], E["image_clip"], g, k=a.k).items()
                            for w in [k.split('_w')[1]]})
best = max(res["rows"], key=lambda n: res["rows"][n]["ndcg"]); res["best"] = best
from stylematch.match import combine
Ebest = E[best] if best in E else combine(E[best.split("+")[0]], E["image_clip"], float(best.split("_w")[1]))
res["rows"][best]["ndcg_ci95_bootstrap"] = R.bootstrap_ndcg(Ebest, g, a.k)
os.makedirs("reports", exist_ok=True); json.dump(res, open(f"reports/shopee_{tag}.json", "w"), indent=1)
for n, v in res["rows"].items(): print(f"{n:36s} NDCG@{a.k}={v['ndcg']:.3f} P@1={v['p_at_1']:.3f} R@{a.k}={v['recall_at_k']:.3f}")
print("best:", best, "95% CI", res["rows"][best]["ndcg_ci95_bootstrap"], "->", f"reports/shopee_{tag}.json")
