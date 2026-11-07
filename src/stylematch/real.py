"""Real-data runs: Shopee (matching on real titles and images) and H&M (attributes and prices on real listings).

Both datasets are Kaggle competition data, licensed for non-commercial and educational use only. They are never
committed: download them yourself into data/raw/ (see README). Every function here is pure so it can be tested on a
tiny synthetic frame with the same schema; the scripts in scripts/ wire them to the real files and write reports/.
"""
from __future__ import annotations
import os, json, hashlib
import numpy as np, pandas as pd

# ----------------------------------------------------------------------------------------------- encoders

def image_encoder(model="ViT-B-32-quickgelu", pretrained="openai", device=None):
    """Return (encode(paths) -> unit vectors, name). Needs `pip install open-clip-torch torch pillow`."""
    import torch, open_clip
    from PIL import Image
    device = device or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    m, _, pre = open_clip.create_model_and_transforms(model, pretrained=pretrained, device=device)
    m.eval()

    def encode(paths, batch=64, log=None):
        out = []
        with torch.no_grad():
            for s in range(0, len(paths), batch):
                ims = [pre(Image.open(p).convert("RGB")) for p in paths[s:s + batch]]
                f = m.encode_image(torch.stack(ims).to(device)).float().cpu().numpy()
                out.append(f)
                if log and (s // batch) % 20 == 0: log(f"  images {s + len(ims)}/{len(paths)}")
        return _norm(np.vstack(out))
    return encode, f"clip:{model}:{pretrained}"


def _norm(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)


def cached(path, fn):
    """np.load(path) if it exists, else fn() saved to path."""
    if os.path.exists(path): return np.load(path)
    x = fn(); os.makedirs(os.path.dirname(path), exist_ok=True); np.save(path, x); return x


# ----------------------------------------------------------------------------------------------- Shopee matching

def shopee_frame(raw_dir, limit=None, seed=0):
    """train.csv: posting_id, image, image_phash, title, label_group. Subsample by whole label groups so every
    retained posting keeps its true matches (otherwise recall would be undefined)."""
    df = pd.read_csv(os.path.join(raw_dir, "train.csv"))
    df["path"] = [os.path.join(raw_dir, "train_images", f) for f in df.image]
    if limit and limit < len(df):
        rng = np.random.default_rng(seed); groups = df.label_group.unique(); rng.shuffle(groups)
        keep, n = [], 0
        for g in groups:
            keep.append(g); n += (df.label_group == g).sum()
            if n >= limit: break
        df = df[df.label_group.isin(keep)]
    return df.reset_index(drop=True)


def evaluate_rows(E_by_name, groups, k=10):
    from stylematch.match import evaluate
    return {name: evaluate(E, groups, k) for name, E in E_by_name.items()}


def fusion_rows(T, I, groups, weights=(0.25, 0.5, 0.75), k=10):
    from stylematch.match import combine, evaluate
    return {f"text_plus_image_w{w}": evaluate(combine(T, I, w), groups, k) for w in weights}


def bootstrap_ndcg(E, groups, k=10, n=200, seed=0):
    """95% interval for NDCG@k by resampling query groups (not rows), so duplicates of a product move together."""
    from stylematch.match import topk
    groups = np.asarray(groups); idx = topk(E, k)
    size = {g: c for g, c in zip(*np.unique(groups, return_counts=True))}
    disc = 1 / np.log2(np.arange(2, k + 2))
    per_group = {}
    for i in range(len(E)):
        m = size[groups[i]] - 1
        if m < 1: continue
        rel = (groups[idx[i]] == groups[i]).astype(float)
        per_group.setdefault(groups[i], []).append((rel * disc).sum() / disc[:min(m, k)].sum())
    gs = list(per_group); vals = [np.mean(per_group[g]) for g in gs]; cnt = [len(per_group[g]) for g in gs]
    rng = np.random.default_rng(seed); means = []
    for _ in range(n):
        b = rng.integers(0, len(gs), len(gs))
        means.append(np.average(np.array(vals)[b], weights=np.array(cnt)[b]))
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


# ----------------------------------------------------------------------------------------------- H&M attributes

HM_TARGETS = {"product_type": "product_type_name", "colour": "colour_group_name", "department": "index_group_name",
              "department_fine": "department_name"}


def hm_articles(raw_dir, min_desc_len=10):
    a = pd.read_csv(os.path.join(raw_dir, "articles.csv"))
    a = a[a.detail_desc.fillna("").str.len() >= min_desc_len].copy()
    a["description"] = a.detail_desc.astype(str)
    return a.reset_index(drop=True)


def hm_attribute_run(a, seed=0, min_class=20, extra_encoder=None):
    """Train on real noisy labels, score on held-out articles split by product_code (colourways of one product never
    straddle the split). Classes with fewer than min_class articles are pooled into 'other'."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupShuffleSplit
    from scipy.sparse import hstack
    from collections import Counter
    tr, te = next(GroupShuffleSplit(1, test_size=.25, random_state=seed).split(a, groups=a.product_code))
    v1 = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    v2 = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=3)
    Xtr = hstack([v1.fit_transform(a.description.iloc[tr]), v2.fit_transform(a.description.iloc[tr])]).tocsr()
    Xte = hstack([v1.transform(a.description.iloc[te]), v2.transform(a.description.iloc[te])]).tocsr()
    if extra_encoder is not None:
        from scipy.sparse import csr_matrix
        Etr = csr_matrix(extra_encoder(list(a.description.iloc[tr]))); Ete = csr_matrix(extra_encoder(list(a.description.iloc[te])))
        Xtr, Xte = hstack([Xtr, Etr]).tocsr(), hstack([Xte, Ete]).tocsr()
    rows, fails = [], {}
    for name, col in HM_TARGETS.items():
        y = a[col].astype(str).copy(); vc = y.iloc[tr].value_counts()
        y = y.where(y.map(vc).fillna(0) >= min_class, "other")
        ytr, yte = y.iloc[tr].values, y.iloc[te].values
        m = LogisticRegression(max_iter=3000, C=5).fit(Xtr, ytr); p = m.predict(Xte)
        maj = Counter(ytr).most_common(1)[0][1] / len(ytr)
        rows.append(dict(attribute=name, column=col, classes=int(len(set(ytr))), accuracy=float((p == yte).mean()),
                         majority_baseline=float(maj), n_test=int(len(yte)), disagreement_rate=float((p != yte).mean())))
        fails[name] = Counter(f"{t} -> {q}" for t, q in zip(yte, p) if t != q).most_common(5)
    return rows, fails, (tr, te)


# ----------------------------------------------------------------------------------------------- H&M prices

def hm_weekly(raw_dir, out_csv, chunksize=2_000_000, log=print):
    """transactions_train.csv (31M rows) -> article-week (units, median price) without loading it whole."""
    if os.path.exists(out_csv): return pd.read_csv(out_csv)
    parts = []
    for i, ch in enumerate(pd.read_csv(os.path.join(raw_dir, "transactions_train.csv"), usecols=["t_dat", "article_id", "price"],
                                       parse_dates=["t_dat"], chunksize=chunksize)):
        ch["week"] = (ch.t_dat - pd.Timestamp("2018-09-17")).dt.days // 7   # first Monday in the data
        parts.append(ch.groupby(["article_id", "week"]).agg(units=("price", "size"), price_sum=("price", "sum"),
                                                              price_med=("price", "median")).reset_index())
        log(f"  chunk {i + 1}: {len(ch):,} rows")
    w = pd.concat(parts).groupby(["article_id", "week"]).agg(units=("units", "sum"), price_sum=("price_sum", "sum"),
                                                              price=("price_med", "median")).reset_index()
    w = w.drop(columns="price_sum"); os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True); w.to_csv(out_csv, index=False); return w


def hm_price_run(weekly, articles, min_weeks=8, min_units=30, seed=0):
    """Reuse the twin's price functions on real H&M article-weeks. H&M prices are scaled (not currency), so only
    relative depth is meaningful. There is no answer key: markdown detection is reported as rates, not precision/recall."""
    from stylematch import price as P
    cnt = weekly.groupby("article_id").agg(weeks=("week", "size"), units=("units", "sum"))
    keep = cnt[(cnt.weeks >= min_weeks) & (cnt.units >= min_units)].index
    tx = weekly[weekly.article_id.isin(keep)]
    rec = P.reconstruct(tx)
    art = articles[["article_id", "product_group_name", "product_type_name"]].rename(columns={"product_group_name": "product_type"})
    art["group"] = articles.product_code.values if "product_code" in articles else art.article_id
    d = rec.merge(art, on="article_id")
    arch = (d.groupby("product_type").agg(articles=("article_id", "count"), median_first_price=("first_price", "median"),
                                          markdown_rate=("md_detected", "mean"),
                                          median_depth=("md_depth_est", lambda s: s[s > 0].median() if (s > 0).any() else np.nan),
                                          median_weeks_to_markdown=("md_week", lambda s: s[s >= 0].median() if (s >= 0).any() else np.nan))
             .round(3).reset_index())
    timing = P.markdown_timing_model(rec, art, seed=seed) if d.md_detected.sum() > 200 else {"skipped": "too few markdowns"}
    wk = P.weekly_by_group(tx, art)
    fc = P.backtest(wk, first_origin=max(60, len(wk) - 44)) if len(wk) >= 70 else {"skipped": "fewer than 70 weeks"}
    return dict(articles_used=int(len(rec)), article_weeks=int(len(tx)), markdown_rate=float(d.md_detected.mean()),
                median_depth_when_marked_down=float(d.loc[d.md_detected, "md_depth_est"].median()) if d.md_detected.any() else None,
                median_weeks_to_first_markdown=float(d.loc[d.md_detected, "md_week"].median()) if d.md_detected.any() else None,
                price_architecture=arch.to_dict("records"), markdown_timing_model=timing, forecast_wape_backtest=fc,
                sellthrough12_by_markdown=rec.groupby("md_detected").sellthrough12.mean().round(3).to_dict()), rec


def file_sha(path, n=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(n))
    return h.hexdigest()[:12]
