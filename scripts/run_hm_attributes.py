"""Attribute extraction on REAL H&M descriptions (Kaggle H&M Personalized Fashion Recommendations; non-commercial).
  PYTHONPATH=src python scripts/run_hm_attributes.py            # TF-IDF + logistic regression
  PYTHONPATH=src python scripts/run_hm_attributes.py --minilm   # adds MiniLM sentence features to the same model"""
import argparse, json, os
from stylematch import real as R
ap = argparse.ArgumentParser(); ap.add_argument("--raw", default="data/raw/hm"); ap.add_argument("--minilm", action="store_true"); ap.add_argument("--limit", type=int)
a = ap.parse_args()
art = R.hm_articles(a.raw)
if a.limit: art = art.sample(a.limit, random_state=0).reset_index(drop=True)
print(f"{len(art):,} articles with a description")
enc = None
if a.minilm:
    from stylematch.match import SentenceTransformerEncoder; st = SentenceTransformerEncoder(); enc = st.transform
rows, fails, _ = R.hm_attribute_run(art, extra_encoder=enc)
res = {"dataset": "H&M articles.csv", "articles": int(len(art)), "features": "tfidf+minilm" if a.minilm else "tfidf",
       "split": "25% held out by product_code", "scores": rows, "top_confusions": fails,
       "note": "No answer key on real data: accuracy is against the catalogue's own labels, which contain unknown error."}
os.makedirs("reports", exist_ok=True); json.dump(res, open(f"reports/hm_attributes_{res['features']}.json", "w"), indent=1)
for r in rows: print(f"{r['attribute']:16s} acc={r['accuracy']:.3f}  majority={r['majority_baseline']:.3f}  classes={r['classes']}")
