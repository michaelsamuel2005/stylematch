import json, sys, numpy as np, pandas as pd
from stylematch.match import TfidfEncoder, combine, evaluate
from stylematch.twin import sim_image_embeddings
art = pd.read_csv("data/articles.csv")
enc_name = sys.argv[1] if len(sys.argv) > 1 else "tfidf"
if enc_name.startswith("st:"):
    from stylematch.match import SentenceTransformerEncoder as Enc; enc = Enc(enc_name[3:])
else: enc = TfidfEncoder()
T = enc.fit_transform(art.description)
I = sim_image_embeddings(art)
res = {"encoder": enc.name, "text_only": evaluate(T, art.group), "image_only_SIMULATED": evaluate(I, art.group)}
for w in (0.25, 0.5):
    res[f"text_plus_image_SIMULATED_w{w}"] = evaluate(combine(T, I, w), art.group)
json.dump(res, open(f"reports/match_{enc.name.replace(':','_')}.json", "w"), indent=1)
for k, v in res.items():
    if isinstance(v, dict): print(f"{k:34s} NDCG@10={v['ndcg']:.3f} P@1={v['p_at_1']:.3f} R@10={v['recall_at_k']:.3f}")
