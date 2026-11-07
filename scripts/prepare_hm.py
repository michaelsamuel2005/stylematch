"""Convert the REAL H&M Personalized Fashion Recommendations files into the stylematch schema (UNTESTED: written without
dataset access). Licence: Kaggle competition data, non-commercial and educational use only. Do not redistribute.
  1. Accept the competition rules on Kaggle and download articles.csv + transactions_train.csv into data/raw/
  2. PYTHONPATH=src python scripts/prepare_hm.py
Real data has NO planted answer key: it is for qualitative checks and the price/forecast parts. Scored evaluation
stays on the simulated twin."""
import pandas as pd
a = pd.read_csv("data/raw/articles.csv")
t = pd.read_csv("data/raw/transactions_train.csv", parse_dates=["t_dat"])
art = pd.DataFrame(dict(article_id=a.article_id, description=a.detail_desc.fillna(""), product_type=a.product_type_name,
                        colour=a.colour_group_name, department=a.department_name, product_group=a.product_group_name))
t["week"] = ((t.t_dat - t.t_dat.min()).dt.days // 7)
tx = t.groupby(["article_id", "week"]).agg(units=("price", "size"), price=("price", "median")).reset_index()
art.to_csv("data/hm_articles.csv", index=False); tx.to_csv("data/hm_transactions.csv", index=False)
print(len(art), "articles;", len(tx), "article-weeks")
# H&M prices are anonymised/scaled; use relative discount depth, not absolute currency values.
