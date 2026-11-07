import json, pandas as pd
from stylematch.attributes import *
art = pd.read_csv("data/articles.csv"); tr, te = split(art)
p = sk_baseline(art, tr, te); rows, fails = score(art, te, p, "tfidf+logreg")
res = {"scores": rows, "top_failures": fails, "error_detection": error_detection(art, te, p),
       "label_noise_in_train": {a: float(art[f"{a}_is_error"].iloc[tr].mean()) for a in ATTRS},
       "transformer": "not run here (needs Hugging Face); see src/stylematch/finetune.py",
       "zero_shot_llm": "not run here (needs API key); see src/stylematch/llm_zero_shot.py"}
json.dump(res, open("reports/attributes.json", "w"), indent=1)
for r in rows: print(r)
print(fails); print(res["error_detection"])
