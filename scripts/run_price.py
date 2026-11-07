import json, pandas as pd
from stylematch.price import *
art = pd.read_csv("data/articles.csv"); tx = pd.read_csv("data/transactions.csv")
rec = reconstruct(tx); rec.to_csv("data/price_history_summary.csv", index=False)
res = {"markdown_detection_vs_planted": score_detection(rec, art),
       "price_architecture": architecture(rec, art).to_dict("records"),
       "markdown_timing_model": markdown_timing_model(rec, art),
       "forecast_wape_backtest": backtest(weekly_by_group(tx, art)),
       "sellthrough12_by_markdown": rec.groupby("md_detected").sellthrough12.mean().round(3).to_dict()}
json.dump(res, open("reports/price.json", "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in res.items() if k != "price_architecture"}, indent=1, default=float))
