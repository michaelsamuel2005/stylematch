"""Fine-tune a small transformer classifier (NOT run in the sandbox: no Hugging Face access here).
Run on your machine: python -m stylematch.finetune product_type   (pip install torch transformers)."""
import sys, pandas as pd, numpy as np


def main(attr, model="distilbert-base-uncased", epochs=3):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from stylematch.attributes import split
    art = pd.read_csv("data/articles.csv"); tr, te = split(art)
    labels = sorted(art[attr].unique()); l2i = {l: i for i, l in enumerate(labels)}
    tok = AutoTokenizer.from_pretrained(model); m = AutoModelForSequenceClassification.from_pretrained(model, num_labels=len(labels))
    dev = "mps" if torch.backends.mps.is_available() else "cpu"; m.to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-5)
    X = list(art.description.iloc[tr]); y = torch.tensor([l2i[v] for v in art[attr].iloc[tr]])
    for ep in range(epochs):
        perm = np.random.permutation(len(X))
        for s in range(0, len(X), 32):
            b = perm[s:s + 32]; enc = tok([X[i] for i in b], padding=True, truncation=True, return_tensors="pt").to(dev)
            loss = m(**enc, labels=y[b].to(dev)).loss; loss.backward(); opt.step(); opt.zero_grad()
    m.eval(); pred = []
    with torch.no_grad():
        for s in range(0, len(te), 64):
            enc = tok(list(art.description.iloc[te][s:s + 64]), padding=True, truncation=True, return_tensors="pt").to(dev)
            pred += m(**enc).logits.argmax(-1).tolist()
    acc = np.mean([labels[p] == t for p, t in zip(pred, art[f"{attr}_true"].iloc[te])]); print(attr, "accuracy vs true labels:", round(float(acc), 4))

if __name__ == "__main__": main(sys.argv[1])
