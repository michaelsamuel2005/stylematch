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

H&M data: Kaggle competition data, non-commercial and educational use only; not included. See `scripts/prepare_hm.py`.
