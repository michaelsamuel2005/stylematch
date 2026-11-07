"""Part 3: price intelligence from article-week transactions (units, price)."""
from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit
from scipy.stats import spearmanr


def reconstruct(tx, tol=0.92, run=2):
    """Per article: first price, markdown week (first of `run` consecutive observed weeks priced <= tol*reference),
    discount depth, time to first markdown (weeks from first sale), and a sell-through proxy."""
    out = []
    for aid, g in tx.sort_values("week").groupby("article_id"):
        w, p, u = g.week.values, g.price.values, g.units.values
        ref = float(np.median(p[:3])); low = p <= tol * ref
        md = None
        for i in range(len(p) - run + 1):
            if low[i:i + run].all(): md = i; break
        first = int(w[0]); cum = np.cumsum(u); tot = cum[-1]
        in12 = u[w < first + 12].sum()
        early4 = u[w < first + 4].sum()
        late = u[w >= w[-1] - 7].sum() if len(w) else 0
        out.append(dict(article_id=aid, first_week=first, first_price=ref, units=int(tot),
                        md_detected=md is not None, md_week=(int(w[md]) - first) if md is not None else -1,
                        md_depth_est=float(1 - np.median(p[md:]) / ref) if md is not None else 0.0,
                        sellthrough12=float(in12 / tot), early4=int(early4), decay_ratio=float((late + 1) / (early4 + 1))))
    return pd.DataFrame(out)


def score_detection(rec, art):
    d = rec.merge(art[["article_id", "md_planted", "md_offset", "md_depth", "launch_week"]], on="article_id")
    tp = (d.md_detected & d.md_planted).sum()
    # only markdowns that happen inside the observation window can be found at all
    m = d[d.md_detected & d.md_planted]
    true_off = m.md_offset + m.launch_week - m.first_week   # planted offset measured from first SALE week
    return dict(precision=float(tp / max(d.md_detected.sum(), 1)), recall=float(tp / max(d.md_planted.sum(), 1)),
                offset_mae_weeks=float((m.md_week - true_off).abs().mean()), depth_mae=float((m.md_depth_est - m.md_depth).abs().mean()),
                n_planted=int(d.md_planted.sum()), n_detected=int(d.md_detected.sum()))


def architecture(rec, art):
    d = rec.merge(art[["article_id", "product_type"]], on="article_id")
    d["tier"] = d.groupby("product_type").first_price.transform(lambda s: pd.qcut(s, 3, labels=["entry", "mid", "premium"]))
    g = d.groupby("product_type").agg(articles=("article_id", "count"), p25=("first_price", lambda s: s.quantile(.25)),
        median=("first_price", "median"), p75=("first_price", lambda s: s.quantile(.75)),
        markdown_rate=("md_detected", "mean"), median_depth=("md_depth_est", lambda s: s[s > 0].median())).round(3)
    return g.reset_index()


def markdown_timing_model(rec, art, seed=0):
    """Gradient boosting for weeks-to-first-markdown among marked-down articles, from signals observable BEFORE markdown
    (type, first price, units in first 4 weeks). Compared with a type-mean baseline on held-out articles."""
    d = rec[rec.md_detected].merge(art[["article_id", "product_type", "group"]], on="article_id")
    d = d[d.md_week >= 4]   # features use weeks 0-3, so only articles still at full price after week 4
    X = pd.get_dummies(d[["product_type"]]).assign(first_price=d.first_price, early4=d.early4)
    y = d.md_week.values
    tr, te = next(GroupShuffleSplit(1, test_size=.3, random_state=seed).split(X, groups=d.group))
    m = GradientBoostingRegressor(random_state=seed, n_estimators=200, max_depth=3, learning_rate=.05).fit(X.iloc[tr], y[tr])
    p = m.predict(X.iloc[te]); base = d.iloc[tr].groupby("product_type").md_week.mean()
    bp = d.iloc[te].product_type.map(base).fillna(y[tr].mean()).values
    return dict(n_train=len(tr), n_test=len(te), mae_gbm=float(np.abs(p - y[te]).mean()), mae_type_mean=float(np.abs(bp - y[te]).mean()),
                spearman=float(spearmanr(p, y[te])[0]),
                importances=dict(sorted(zip(X.columns, m.feature_importances_.round(3)), key=lambda t: -t[1])[:5]))


def weekly_by_group(tx, art):
    d = tx.merge(art[["article_id", "product_type"]], on="article_id")
    return d.pivot_table(index="week", columns="product_type", values="units", aggfunc="sum", fill_value=0)


def backtest(series, horizon=4, first_origin=60, step=4):
    """Rolling-origin backtest per group. Models: last-4-week mean, seasonal naive (52w lag; needs >=52 weeks of history),
    and ridge on trend + Fourier terms. WAPE pooled over all origins/horizons."""
    from sklearn.linear_model import Ridge
    res = {k: [0.0, 0.0] for k in ("naive_last4", "seasonal_naive", "ridge_fourier")}
    for col in series.columns:
        y = series[col].values.astype(float); T = len(y)
        for o in range(first_origin, T - horizon + 1, step):
            act = y[o:o + horizon]; hist = y[:o]
            preds = {"naive_last4": np.repeat(hist[-4:].mean(), horizon),
                     "seasonal_naive": y[o - 52:o - 52 + horizon]}
            t = np.arange(o + horizon); F = np.column_stack([t] + [f(2 * np.pi * k * t / 52) for k in (1, 2) for f in (np.sin, np.cos)])
            r = Ridge(alpha=1.0).fit(F[:o], hist); preds["ridge_fourier"] = np.clip(r.predict(F[o:o + horizon]), 0, None)
            for k, p in preds.items(): res[k][0] += np.abs(p - act).sum(); res[k][1] += act.sum()
    return {k: round(a / b, 4) for k, (a, b) in res.items()}
