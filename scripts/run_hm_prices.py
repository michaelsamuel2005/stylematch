"""Price intelligence on REAL H&M transactions (Kaggle; non-commercial). Prices in this dataset are scaled, so
depth is relative. First run aggregates the 3.5 GB transactions file to article-weeks (takes several minutes; cached).
  PYTHONPATH=src python scripts/run_hm_prices.py"""
import argparse, json, os, pandas as pd
from stylematch import real as R
ap = argparse.ArgumentParser(); ap.add_argument("--raw", default="data/raw/hm"); a = ap.parse_args()
weekly = R.hm_weekly(a.raw, "data/cache/hm_weekly.csv")
art = pd.read_csv(os.path.join(a.raw, "articles.csv"))
res, rec = R.hm_price_run(weekly, art)
res["dataset"] = "H&M transactions_train.csv aggregated to article-week"
os.makedirs("reports", exist_ok=True); json.dump(res, open("reports/hm_prices.json", "w"), indent=1, default=float)
rec.to_csv("data/cache/hm_price_history_summary.csv", index=False)
print(json.dumps({k: v for k, v in res.items() if k != "price_architecture"}, indent=1, default=float))
print(pd.DataFrame(res["price_architecture"]).to_string())
