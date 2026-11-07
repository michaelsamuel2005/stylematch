"""The real-data code paths on tiny synthetic frames in the Kaggle schemas (the data itself is never in the repo)."""
import os, numpy as np, pandas as pd, pytest
from stylematch import real as R

def _shopee(tmp_path, n_groups=30):
    from PIL import Image
    raw = tmp_path / "shopee"; (raw / "train_images").mkdir(parents=True)
    rng = np.random.default_rng(0); rows = []
    for g in range(n_groups):
        words = " ".join(rng.choice(list("abcdefghijklmnopqrstuvwxyz"), 4)) + f" item{g}"
        for j in range(int(rng.integers(1, 4))):
            fn = f"{g}_{j}.jpg"; Image.new("RGB", (8, 8), (g * 7 % 255, 0, 0)).save(raw / "train_images" / fn)
            rows.append(dict(posting_id=f"p{g}_{j}", image=fn, image_phash="x", title=f"{words} {'xl' if j else ''}", label_group=1000 + g))
    pd.DataFrame(rows).to_csv(raw / "train.csv", index=False); return str(raw)

def test_shopee_frame_subsamples_whole_groups(tmp_path):
    raw = _shopee(tmp_path); df = R.shopee_frame(raw, limit=20)
    full = pd.read_csv(os.path.join(raw, "train.csv"))
    for g, c in df.label_group.value_counts().items(): assert c == (full.label_group == g).sum()

def test_shopee_text_pipeline_and_bootstrap(tmp_path):
    from stylematch.match import TfidfEncoder
    df = R.shopee_frame(_shopee(tmp_path)); E = TfidfEncoder(32).fit_transform(df.title)
    rows = R.evaluate_rows({"t": E}, df.label_group.values, 5); assert rows["t"]["ndcg"] > 0.5
    lo, hi = R.bootstrap_ndcg(E, df.label_group.values, 5, n=30); assert 0 <= lo <= hi <= 1
    I = np.random.default_rng(1).normal(size=(len(df), 8)); I /= np.linalg.norm(I, axis=1, keepdims=True)
    assert set(R.fusion_rows(E, I, df.label_group.values)) == {"text_plus_image_w0.25", "text_plus_image_w0.5", "text_plus_image_w0.75"}

def _hm_articles(n=400):
    rng = np.random.default_rng(0); types = ["Trousers", "Sweater", "T-shirt", "Dress"]; cols = ["Black", "White", "Blue"]
    rows = []
    for i in range(n):
        t, c = rng.choice(types), rng.choice(cols); grp = "Ladieswear" if rng.random() < .6 else "Menswear"
        rows.append(dict(article_id=100000 + i, product_code=100000 + i // 3, product_type_name=t, colour_group_name=c,
                         index_group_name=grp, department_name=f"{t} dept", product_group_name="Garment",
                         detail_desc=f"{c.lower()} {t.lower()} in soft cotton for {grp.lower()} with a relaxed fit"))
    return pd.DataFrame(rows)

def test_hm_attributes_split_by_product_code_and_scores():
    a = _hm_articles(); a["description"] = a.detail_desc
    rows, fails, (tr, te) = R.hm_attribute_run(a, min_class=5)
    assert not set(a.product_code.iloc[tr]) & set(a.product_code.iloc[te])
    acc = {r["attribute"]: r["accuracy"] for r in rows}; assert acc["product_type"] > 0.9 and acc["colour"] > 0.9

def test_hm_weekly_and_price_run(tmp_path):
    raw = tmp_path / "hm"; raw.mkdir(); a = _hm_articles(60); a.to_csv(raw / "articles.csv", index=False)
    rng = np.random.default_rng(0); rows = []
    for aid in a.article_id:
        base = rng.uniform(.01, .1); start = int(rng.integers(0, 20))
        for w in range(start, start + 30):
            price = base * (0.6 if w >= start + 12 else 1.0)
            for _ in range(int(rng.integers(1, 6))):
                rows.append(dict(t_dat=pd.Timestamp("2018-09-17") + pd.Timedelta(days=7 * w + int(rng.integers(0, 7))), customer_id="c", article_id=aid, price=round(price, 4), sales_channel_id=2))
    pd.DataFrame(rows).to_csv(raw / "transactions_train.csv", index=False)
    weekly = R.hm_weekly(str(raw), str(tmp_path / "weekly.csv"), chunksize=1000, log=lambda s: None)
    assert set(weekly.columns) == {"article_id", "week", "units", "price"} and weekly.units.sum() == len(rows)
    res, rec = R.hm_price_run(weekly, a, min_weeks=8, min_units=10)
    assert res["articles_used"] == 60 and 0.9 < res["markdown_rate"] <= 1 and abs(res["median_depth_when_marked_down"] - 0.4) < 0.05
