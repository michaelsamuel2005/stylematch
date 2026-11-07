# stylematch

Product matching, attribute extraction and price intelligence for fashion retail, scored against a planted answer key, with a matcher served over FastAPI.

**Status:** every number in this repo comes from a **simulated twin** (labelled as such). Sentence-transformer, CLIP, fine-tuned transformer and zero-shot LLM runs are implemented as optional one-command runs but **have not been run**; the real H&M path is untested. See [MODEL_CARD.md](MODEL_CARD.md).

```bash
pip install -r requirements.txt
make run      # twin -> match, attributes, price reports in reports/
make test     # 9 tests
make serve    # POST /match {"description": "...", "k": 5}
docker build -t stylematch . && docker run -p 8000:8000 stylematch
```

| Part | Code | Report |
|---|---|---|
| 1 Matching: NDCG/precision/recall@k vs planted groups, text vs text+image | `match.py` | `reports/match_*.json` |
| 2 Attributes: sklearn baseline; transformer and zero-shot LLM provided, not run | `attributes.py`, `finetune.py`, `llm_zero_shot.py` | `reports/attributes.json` |
| 3 Prices: markdown detection, architecture, timing model, backtested forecast | `price.py` | `reports/price.json` |
| 4 Serving: FastAPI, Docker, CI | `api.py`, `Dockerfile`, `.github/workflows/ci.yml` | |

Optional real-model runs (need Hugging Face access): `pip install -r requirements-extra.txt`, then `python scripts/run_match.py st:sentence-transformers/all-MiniLM-L6-v2`, `python -m stylematch.finetune department`.

## Real data (Kaggle; non-commercial and educational use only; never committed)

| Dataset | Used for | Download to |
|---|---|---|
| [Shopee Price Match Guarantee](https://www.kaggle.com/competitions/shopee-product-matching) | matching: real titles + images, labelled match groups | `data/raw/shopee/` (`train.csv`, `train_images/`) |
| [H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) | attributes (`articles.csv`), prices (`transactions_train.csv`) | `data/raw/hm/` |

```bash
pip install -r requirements-extra.txt                      # torch, open-clip, sentence-transformers
PYTHONPATH=src python scripts/run_shopee.py --limit 3000   # quick check, then drop --limit for the full 34k
PYTHONPATH=src python scripts/run_hm_attributes.py
PYTHONPATH=src python scripts/run_hm_prices.py             # aggregates 31M transactions once, cached
```
Accept each competition's rules on Kaggle before downloading. Results land in `reports/` and are copied into the model card by hand.
