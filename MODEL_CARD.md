# stylematch model card

**What this is.** A portfolio project that mirrors three retail-intelligence problems (product matching, attribute extraction, price intelligence) on a **simulated twin** of a fashion catalogue. It is not a production system and makes no claim about any company's data.

## Data
- **Scored results use a simulated twin** (`src/stylematch/twin.py`): 4,584 articles in 3,000 product groups (1,584 planted near-duplicates), 165,587 article-week sales rows over 104 weeks. Defects are planted with a recorded answer key: duplicates (other colourway or department, reworded), markdowns (week and depth), and wrong labels (about 6% per attribute).
- **H&M Personalized Fashion Recommendations** (Kaggle) is supported through `scripts/prepare_hm.py`, for **non-commercial and educational use only**; the data is not in this repo. That script was written without dataset access and has not been run. No result below comes from H&M data.
- "Image" results use a **simulated stand-in vector**, not CLIP. CLIP on real images has not been run.

## Results actually produced (seed 7, `make run`)
| Part | Method | Result |
|---|---|---|
| Match | TF-IDF (char n-gram) + SVD, text only | NDCG@10 0.700, P@1 0.597, recall@10 0.838 |
| Match | + simulated image vector, weight 0.25 | NDCG@10 0.710, P@1 0.603 (a +0.010 NDCG change, one seed, no confidence interval) |
| Match | + simulated image vector, weight 0.5 | NDCG@10 0.641 (worse) |
| Match | MiniLM-L6-v2 sentence embeddings, text only (run on a Mac, same twin) | NDCG@10 0.409, P@1 0.322, recall@10 0.539: worse than TF-IDF (0.700). Likely cause: the twin's identifying signal is rare lexical detail words, which TF-IDF matches exactly; untested. |
| Attributes | TF-IDF + logistic regression, scored on true labels | product type 99.8%, colour 91.6%, department 75.7% |
| Prices | markdown detection vs planted | precision 1.00, recall 0.89, offset error 0.38 weeks, depth error 0.003 |
| Prices | time-to-markdown, gradient boosting vs type mean | MAE 3.6 vs 5.7 weeks |
| Forecast | rolling-origin WAPE, 4-week horizon | last-4-week mean 0.115, ridge with Fourier terms 0.270, seasonal naive 0.463 |

Not run (code present, no numbers claimed): CLIP, fine-tuned transformer, zero-shot LLM.

## Real data (code written, results pending)
The twin results above partly reflect how the twin was generated. `scripts/run_shopee.py` (real titles and images with labelled match groups), `scripts/run_hm_attributes.py` and `scripts/run_hm_prices.py` run the same evaluation harnesses on Kaggle's Shopee Price Match Guarantee and H&M datasets, both non-commercial and never committed. This section is filled in from `reports/shopee_*.json`, `reports/hm_attributes_*.json` and `reports/hm_prices.json` once they exist; until then no real-data claim is made.

## Where it fails
- Department is the weakest attribute: 235 of the errors are Ladieswear/Menswear swaps, because many descriptions never say who the item is for. Flagging label disagreements finds nearly all planted department errors (recall 0.95) but with precision 0.19, so it is a review queue, not an auto-fix. Colour flagging has precision 0.41 for the same reason.
- Product type is near-trivial in the twin because nouns are type-specific. Real catalogues are messier, so 99.8% should not be read as a benchmark.
- The markdown timing model learns that slow starters are marked down sooner, **a dependency I planted in the simulator**. It shows the pipeline recovers a known effect, not that real retailers behave this way.
- The seasonal-naive forecast loses badly because the twin's articles launch and decay within the window; the simple recent-mean wins. This is a property of the twin.
- Text-plus-image gains are within what one seed can produce; treat the fusion comparison as a test of the evaluation harness only.

## What the data cannot support
- Real-world accuracy, real markdown behaviour, or any comparison with commercial matching products. Retailer websites, crawling, competitor prices and stock-outs are absent; sales are used only as a sell-through proxy, not true sell-out.
- Elasticity or causal claims: the twin's price response is a modelling assumption.
- Image conclusions: no real images were processed.
