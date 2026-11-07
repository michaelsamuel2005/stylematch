"""Simulated retail twin with an answer key.

Everything here is SIMULATED. It mimics the shape of a fashion catalogue
(descriptions, type/colour/department labels, weekly prices and sales) and
plants three kinds of defects whose truth is recorded, so every method is
scored against a known answer.
"""
from __future__ import annotations
import numpy as np, pandas as pd

TYPES = {  # type -> (departments, nouns, fabrics)
    "Trousers": (["Ladieswear", "Menswear"], ["trousers", "chinos", "joggers"], ["cotton twill", "stretch denim", "linen blend"]),
    "Dress": (["Ladieswear"], ["dress", "midi dress", "wrap dress"], ["viscose", "jersey", "satin"]),
    "Sweater": (["Ladieswear", "Menswear"], ["jumper", "sweater", "knit"], ["merino wool", "cotton knit", "fine knit"]),
    "T-shirt": (["Ladieswear", "Menswear", "Kids"], ["top", "t-shirt", "tee"], ["soft jersey", "organic cotton", "slub cotton"]),
    "Shoes": (["Ladieswear", "Menswear"], ["trainers", "boots", "loafers"], ["leather", "canvas", "suede"]),
    "Coat": (["Ladieswear", "Menswear"], ["coat", "parka", "jacket"], ["wool blend", "water-repellent shell", "quilted fabric"]),
    "Skirt": (["Ladieswear"], ["skirt", "pleated skirt", "mini skirt"], ["crepe", "denim", "satin"]),
    "Swimwear": (["Ladieswear", "Menswear"], ["swimsuit", "swim shorts", "bikini top"], ["quick-dry fabric", "recycled nylon"]),
}
COLOURS = ["Black", "White", "Navy", "Beige", "Red", "Green", "Blue", "Grey", "Pink", "Brown"]
FEATS = ["with an elasticated waist", "with a ribbed hem", "with pockets", "with a relaxed fit",
         "with a slim fit", "with a button fastening", "with a zip fastening", "with a lined interior",
         "in a regular fit", "with a drawstring"]
DETAIL = ("embroidered floral stripe herringbone cable striped checked pleated ruched quilted bonded textured "
          "tapered cropped oversized longline belted hooded collared tiered smocked gathered asymmetric "
          "paisley houndstooth gingham polka ribbed waffle fringed studded lace mesh cutout flared").split()
SYN = {"trousers": "pants", "top": "shirt", "jumper": "pullover", "trainers": "sneakers", "coat": "overcoat",
       "dress": "frock", "tee": "t shirt", "knit": "knitted top", "skirt": "skirt", "swimsuit": "bathing suit"}
SEASON_AMP = {"Swimwear": 0.8, "Coat": 0.7, "Sweater": 0.4, "Dress": 0.4}


def _desc(rng, typ, noun, fabric, feat, colour, dept, detail=""):
    return f"{colour.lower()} {detail} {noun} in {fabric} {feat}." + (f" {dept} {typ.lower()}." if rng.random() < .4 else "")


def _reword(rng, noun, fabric, feat, colour, dept, typ, detail=""):
    n = SYN.get(noun, noun)
    dw = detail.split(); dw = [w for w in dw if rng.random() < .85] or dw[:1]; n = " ".join(dw) + " " + n
    f2 = rng.choice(FEATS)
    parts = [f"{n} made of {fabric}", f"{f2 if rng.random() < .3 else feat}"]
    rng.shuffle(parts)
    c = colour.lower() if rng.random() < .7 else ""
    return (f"{c} {' '.join(parts)}." + (f" {typ.lower()} for {dept.lower()}." if rng.random() < .4 else "")).strip()


def make_twin(n_base=3000, weeks=104, seed=7, dup_rate=0.35, err_rate=0.06, md_rate=0.55):
    rng = np.random.default_rng(seed)
    rows, gid = [], 0
    for _ in range(n_base):
        typ = rng.choice(list(TYPES)); depts, nouns, fabs = TYPES[typ]
        noun, fabric, feat = rng.choice(nouns), rng.choice(fabs), rng.choice(FEATS)
        dept, colour = rng.choice(depts), rng.choice(COLOURS)
        det = " ".join(rng.choice(DETAIL, 2, replace=False)); rows.append(dict(group=gid, base=1, detail=det, product_type=typ, colour=colour, department=dept,
                         noun=noun, fabric=fabric, feat=feat))
        if rng.random() < dup_rate:  # planted near-duplicate: other colourway and/or department, reworded
            for _k in range(int(rng.integers(1, 3))):
                c2 = rng.choice([c for c in COLOURS if c != colour]) if rng.random() < .8 else colour
                d2 = rng.choice(depts)
                rows.append(dict(group=gid, base=0, detail=det, product_type=typ, colour=c2, department=d2,
                                 noun=noun, fabric=fabric, feat=feat))
        gid += 1
    art = pd.DataFrame(rows).reset_index(drop=True)
    art.insert(0, "article_id", np.arange(len(art)) + 100000)
    reword = np.array([not b or rng.random() < .0 for b in art.base])
    art["description"] = [
        _reword(rng, r.noun, r.fabric, r.feat, r.colour, r.department, r.product_type, r.detail) if rw
        else _desc(rng, r.product_type, r.noun, r.fabric, r.feat, r.colour, r.department, r.detail)
        for r, rw in zip(art.itertuples(), reword)]
    # planted attribute errors: label disagrees with the description (truth kept)
    for col, pool in [("product_type", list(TYPES)), ("colour", COLOURS), ("department", ["Ladieswear", "Menswear", "Kids"])]:
        art[f"{col}_true"] = art[col]
        bad = rng.random(len(art)) < err_rate
        art.loc[bad, col] = [rng.choice([p for p in pool if p != v]) for v in art.loc[bad, col]]
        art[f"{col}_is_error"] = bad
    art["base_price"] = np.round(rng.lognormal(np.log(25), .6, len(art)).clip(5, 250), 2)
    art["launch_week"] = rng.integers(0, weeks - 20, len(art))
    # planted markdowns: week offset and depth are the truth
    has = rng.random(len(art)) < md_rate
    art["md_planted"] = has
    pop = rng.gamma(1.2, 2.0, len(art))
    pct = pd.Series(pop).rank(pct=True).values   # slow sellers are marked down earlier (planted dependence)
    off = np.clip(np.round(5 + 24 * pct + rng.normal(0, 2.5, len(art))), 3, 34).astype(int)
    art["md_offset"] = np.where(has, off, -1)
    art["md_depth"] = np.where(has, rng.choice([.2, .3, .4, .5], len(art)), 0.0)
    # transactions (weekly, aggregated: units and revenue per article-week)
    wk = np.arange(weeks)
    tx = []
    amp = art.product_type.map(SEASON_AMP).fillna(.15).values
    ph = art.product_type.map(lambda t: 26 if t in ("Swimwear",) else 0).values
    for i, a in enumerate(art.itertuples()):
        w = wk[wk >= a.launch_week]; age = w - a.launch_week
        season = 1 + amp[i] * np.sin(2 * np.pi * (w - ph[i]) / 52)
        md_on = a.md_planted & (age >= a.md_offset)
        price = a.base_price * np.where(md_on, 1 - a.md_depth, 1.0)
        elast = np.where(md_on, 1 + 2.2 * a.md_depth, 1.0)
        lam = pop[i] * season * np.exp(-age / 45.0) * elast
        # mild non-strict noise: observed price jitter (promo noise, ±2%)
        units = rng.poisson(lam)
        k = units > 0
        pr = np.round(price * rng.choice([1, 1, 1, .98, 1.02], len(w)), 2)
        tx.append(pd.DataFrame(dict(article_id=a.article_id, week=w[k], units=units[k], price=pr[k])))
    tx = pd.concat(tx, ignore_index=True)
    # simulated image stand-in: latent visual vector from (type, noun, fabric) + colour, plus noise.
    # NOT a real CLIP embedding; used only to exercise the text-vs-text+image harness.
    return art, tx


def sim_image_embeddings(art, dim=32, noise=0.5, seed=3):
    rng = np.random.default_rng(seed)
    cache = {}
    def vec(k):
        if k not in cache:
            cache[k] = rng.normal(size=dim)
        return cache[k]
    out = np.stack([vec(("s", r.product_type, r.noun, r.fabric)) + .3 * vec(("c", r.colour)) for r in art.itertuples()])
    out = out + rng.normal(scale=noise, size=out.shape)
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def save(art, tx, d="data"):
    import os; os.makedirs(d, exist_ok=True)
    art.to_csv(f"{d}/articles.csv", index=False); tx.to_csv(f"{d}/transactions.csv", index=False)

if __name__ == "__main__":
    a, t = make_twin(); save(a, t); print(len(a), len(t), a.group.nunique())
