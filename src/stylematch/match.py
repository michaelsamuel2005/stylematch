"""Part 1: product matching. Encoders are pluggable; evaluation is against planted groups."""
from __future__ import annotations
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


class TfidfEncoder:
    """Local, dependency-light text encoder (word+char TF-IDF -> SVD). Baseline, not a transformer."""
    name = "tfidf-svd"
    def __init__(self, dim=128): self.dim = dim
    def fit_transform(self, texts):
        self.v = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2)
        X = self.v.fit_transform(texts)
        self.svd = TruncatedSVD(min(self.dim, X.shape[1] - 1), random_state=0)
        return _norm(self.svd.fit_transform(X))
    def transform(self, texts): return _norm(self.svd.transform(self.v.transform(texts)))


class SentenceTransformerEncoder:
    """Real sentence-transformer. Needs `pip install sentence-transformers` and Hugging Face access."""
    def __init__(self, model="sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(model); self.name = f"st:{model.split('/')[-1]}"
    def fit_transform(self, texts): return self.transform(texts)
    def transform(self, texts): return _norm(self.m.encode(list(texts), batch_size=64, show_progress_bar=False))


def _norm(x): return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)


def combine(text_emb, img_emb, w_img=0.5):
    """Late fusion: weighted concatenation of unit vectors (cosine = weighted sum of cosines)."""
    return _norm(np.hstack([np.sqrt(1 - w_img) * text_emb, np.sqrt(w_img) * img_emb]))


def topk(E, k=10, block=1024):
    n = len(E); idx = np.zeros((n, k), int)
    for s in range(0, n, block):
        S = E[s:s + block] @ E.T
        S[np.arange(S.shape[0]), np.arange(s, s + S.shape[0])] = -9
        idx[s:s + block] = np.argsort(-S, axis=1)[:, :k]
    return idx


def evaluate(E, groups, k=10):
    """NDCG@k, precision@k, recall@k over queries whose group has >=2 members (binary relevance)."""
    groups = np.asarray(groups); idx = topk(E, k)
    size = {g: c for g, c in zip(*np.unique(groups, return_counts=True))}
    disc = 1 / np.log2(np.arange(2, k + 2)); nd, p1, pk, rk = [], [], [], []
    for i in range(len(E)):
        m = size[groups[i]] - 1
        if m < 1: continue
        rel = (groups[idx[i]] == groups[i]).astype(float)
        ideal = disc[:min(m, k)].sum()
        nd.append((rel * disc).sum() / ideal); p1.append(rel[0]); pk.append(rel.sum() / k); rk.append(rel.sum() / m)
    return dict(queries=len(nd), ndcg=float(np.mean(nd)), p_at_1=float(np.mean(p1)),
                p_at_k=float(np.mean(pk)), recall_at_k=float(np.mean(rk)), k=k)
