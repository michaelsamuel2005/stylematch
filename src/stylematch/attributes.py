"""Part 2: attribute extraction from the description alone.
Labels in the catalogue contain planted errors; models train on the noisy labels and are scored on the TRUE labels."""
from __future__ import annotations
import numpy as np, pandas as pd
from collections import Counter
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit

ATTRS = ["product_type", "colour", "department"]


def split(art, seed=0):
    tr, te = next(GroupShuffleSplit(1, test_size=.25, random_state=seed).split(art, groups=art.group))
    return tr, te


def sk_baseline(art, tr, te):
    v1 = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True); v2 = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2)
    Xtr = hstack([v1.fit_transform(art.description.iloc[tr]), v2.fit_transform(art.description.iloc[tr])]).tocsr()
    Xte = hstack([v1.transform(art.description.iloc[te]), v2.transform(art.description.iloc[te])]).tocsr()
    out = {}
    for a in ATTRS:
        m = LogisticRegression(max_iter=2000, C=10).fit(Xtr, art[a].iloc[tr]); out[a] = m.predict(Xte)
    return out


def score(art, te, preds, name):
    rows, fails = [], {}
    for a in ATTRS:
        y = art[f"{a}_true"].iloc[te].values; p = np.asarray(preds[a])
        rows.append(dict(method=name, attribute=a, accuracy=float((y == p).mean()), n=len(y)))
        fails[a] = Counter(f"{t} -> {q}" for t, q in zip(y, p) if t != q).most_common(5)
    return rows, fails


def error_detection(art, te, preds):
    """Flag a catalogue label as suspect when the description-based prediction disagrees; score vs planted errors."""
    res = []
    for a in ATTRS:
        flag = np.asarray(preds[a]) != art[a].iloc[te].values; truth = art[f"{a}_is_error"].iloc[te].values
        tp = (flag & truth).sum()
        res.append(dict(attribute=a, precision=float(tp / max(flag.sum(), 1)), recall=float(tp / max(truth.sum(), 1)), planted=int(truth.sum())))
    return res


def rule_zero_shot_stub(art, te):
    """NOT an LLM. A keyword lookup used only to test the harness plumbing in CI."""
    kw = {"product_type": {"trousers": "Trousers", "pants": "Trousers", "chinos": "Trousers", "dress": "Dress", "frock": "Dress"}}
    return {a: [next((v for k, v in kw.get(a, {}).items() if k in d), "unknown") for d in art.description.iloc[te]] for a in ATTRS}
