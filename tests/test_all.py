import numpy as np, pandas as pd, pytest
from stylematch.twin import make_twin, sim_image_embeddings
from stylematch.match import TfidfEncoder, evaluate, combine, topk
from stylematch import attributes as A, price as P

@pytest.fixture(scope="module")
def twin(): return make_twin(n_base=400, weeks=104, seed=1)

def test_twin_deterministic_and_keyed(twin):
    a2, _ = make_twin(n_base=400, weeks=104, seed=1)
    assert a2.description.equals(twin[0].description)
    art = twin[0]; assert art.group.nunique() == 400 and (art.groupby("group").size() > 1).any()
    assert art.product_type_is_error.any() and art.md_planted.any()

def test_match_beats_chance_and_metrics_bounded(twin):
    art = twin[0]; E = TfidfEncoder(64).fit_transform(art.description)
    r = evaluate(E, art.group, 10); assert 0 <= r["ndcg"] <= 1 and r["ndcg"] > 0.3
    rnd = evaluate(np.random.default_rng(0).normal(size=E.shape), art.group, 10); assert r["ndcg"] > 5 * rnd["ndcg"]

def test_ndcg_perfect_when_group_embeddings_identical():
    g = np.repeat(np.arange(20), 2); E = np.eye(20)[g]
    assert evaluate(E, g, 5)["ndcg"] == pytest.approx(1.0)

def test_fusion_is_unit_norm(twin):
    art = twin[0]; T = TfidfEncoder(32).fit_transform(art.description)
    assert np.allclose(np.linalg.norm(combine(T, sim_image_embeddings(art), .5), axis=1), 1)

def test_attributes_beat_majority(twin):
    art = twin[0]; tr, te = A.split(art); p = A.sk_baseline(art, tr, te)
    rows, _ = A.score(art, te, p, "x")
    acc = {r["attribute"]: r["accuracy"] for r in rows}
    assert acc["product_type"] > 0.9 and acc["colour"] > 0.6

def test_markdown_detection_vs_planted(twin):
    art, tx = twin; rec = P.reconstruct(tx); s = P.score_detection(rec, art)
    assert s["precision"] > 0.95 and s["recall"] > 0.7 and s["offset_mae_weeks"] < 2

def test_no_markdown_flagged_on_flat_prices():
    tx = pd.DataFrame(dict(article_id=1, week=range(20), units=3, price=20.0)); assert not P.reconstruct(tx).md_detected.iloc[0]

def test_backtest_runs(twin):
    r = P.backtest(P.weekly_by_group(twin[1], twin[0])); assert set(r) == {"naive_last4", "seasonal_naive", "ridge_fourier"}

def test_api(twin, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from stylematch import api
    p = tmp_path / "a.csv"; twin[0].to_csv(p, index=False); api.load(str(p)); c = TestClient(api.app)
    assert c.get("/health").json()["articles"] == len(twin[0])
    d = twin[0].description.iloc[0]; r = c.post("/match", json={"description": d, "k": 3}).json()["matches"]
    assert len(r) == 3 and r[0]["score"] > 0.9
    assert c.post("/match", json={"description": "x", "k": 3}).status_code == 422
