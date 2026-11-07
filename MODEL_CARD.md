# stylematch model card

**What this is.** A portfolio project that mirrors three retail-intelligence problems (product matching, attribute extraction, price intelligence). Matching is evaluated on **real product listings**; attributes and prices are evaluated on a **simulated twin** with a planted answer key, with real-data runs for H&M written and pending. It is not a production system and makes no claim about any company's data.

## Data
- **Real listings (matching): Shopee Price Match Guarantee** (Kaggle, train split; non-commercial and educational use only, never committed). 34,250 product postings with a title and an image, in 11,014 labelled groups of the same product. Every posting has at least one true match. Scored with `scripts/run_shopee.py`; the report is `reports/shopee_n34250.json`.
- **Simulated twin (attributes, prices, the API's index):** `src/stylematch/twin.py`, 4,584 articles in 3,000 product groups (1,584 planted near-duplicates), 165,587 article-week sales rows over 104 weeks. Defects are planted with a recorded answer key: duplicates (other colourway or department, reworded), markdowns (week and depth), and wrong labels (about 6% per attribute).
- **H&M Personalized Fashion Recommendations** (Kaggle; non-commercial): `scripts/run_hm_attributes.py` and `scripts/run_hm_prices.py` are written and tested on synthetic frames in the H&M schema. No H&M result is claimed until their reports are committed.

## Matching on real listings (Shopee, all 34,250 postings)
Zero-shot embeddings, nothing trained on Shopee. Every posting is a query against all others; relevance is "same label group".

| Method | NDCG@10 | Precision@1 | Recall@10 |
|---|---:|---:|---:|
| MiniLM-L6-v2 title + CLIP ViT-B/32 image, equal weight | **0.766** (95% interval 0.760 to 0.772) | **0.743** | **0.788** |
| TF-IDF title + CLIP image, image weight 0.75 | 0.740 | 0.720 | 0.762 |
| MiniLM-L6-v2 title only | 0.639 | 0.631 | 0.661 |
| CLIP image only | 0.602 | 0.619 | 0.603 |
| TF-IDF (character n-grams + SVD) title only | 0.543 | 0.539 | 0.568 |

- **Text and image are complementary.** Fusion adds 0.127 NDCG@10 over the best single signal. A likely reason, not yet tested: titles separate products whose photos look alike, and photos match products whose sellers wrote different titles.
- **The simulated twin and a small sample both pointed the wrong way.** On the twin, TF-IDF beat MiniLM (0.700 vs 0.409), and on a 3,001-posting sample of Shopee TF-IDF still led (0.860 vs 0.819). On all 34,250 postings MiniLM wins clearly (0.639 vs 0.543). With ten times more distractors, near-identical character strings stop being enough. The twin's design (rare detail words identify a product) favoured TF-IDF; that is why the real-data run exists.
- **Scale changes the numbers.** The 3,001-posting sample scored NDCG@10 0.918 for its best fusion; the full set scores 0.766 for its best. Only full-scale numbers should be quoted.
- **Caveat on the fusion weight.** Three weights (0.25, 0.5, 0.75) were compared on the same postings they are scored on, so the best row is slightly optimistic. The interval comes from resampling whole label groups and does not cover that choice; a held-out split for the weight is the next step.
- The 3,001-posting sample ran before the CLIP configuration fix (quick-GELU), so its image scores are not directly comparable with the full run.

## Twin results (seed 7, `make run`)
| Part | Method | Result |
|---|---|---|
| Match | TF-IDF (char n-gram) + SVD, text only | NDCG@10 0.700, P@1 0.597, recall@10 0.838 |
| Match | MiniLM-L6-v2, text only | NDCG@10 0.409, P@1 0.322, recall@10 0.539 |
| Match | TF-IDF + simulated image vector, weight 0.25 | NDCG@10 0.710 (the vector is a stand-in, not CLIP) |
| Attributes | TF-IDF + logistic regression, scored on true labels | product type 99.8%, colour 91.6%, department 75.7% |
| Prices | markdown detection vs planted | precision 1.00, recall 0.89, offset error 0.38 weeks, depth error 0.003 |
| Prices | time-to-markdown, gradient boosting vs type mean | MAE 3.6 vs 5.7 weeks |
| Forecast | rolling-origin WAPE, 4-week horizon | last-4-week mean 0.115, ridge with Fourier terms 0.270, seasonal naive 0.463 |

Not run (code present, no numbers claimed): fine-tuned transformer, zero-shot LLM, H&M attributes and prices.

## Where it fails
- Matching: about a quarter of real postings do not get their true match ranked first, even with fusion. The model is zero-shot; fine-tuning the encoders on matched pairs, or adding price and phash features, is the obvious next step.
- Department is the weakest attribute on the twin: 235 of the errors are Ladieswear/Menswear swaps, because many descriptions never say who the item is for. Flagging label disagreements finds nearly all planted department errors (recall 0.95) but with precision 0.19, so it is a review queue, not an auto-fix.
- Product type is near-trivial in the twin because nouns are type-specific. 99.8% is not a benchmark.
- The markdown timing model learns that slow starters are marked down sooner, **a dependency planted in the simulator**. It shows the pipeline recovers a known effect, not that real retailers behave this way.
- The seasonal-naive forecast loses because the twin's articles launch and decay within the window; this is a property of the twin.

## What the data cannot support
- Comparisons with commercial matching products, or accuracy on another retailer's catalogue: Shopee is one marketplace, mostly in Indonesian and English, and its labels are the competition's.
- Real markdown behaviour, elasticity or causal price claims: the price results are on the twin until the H&M run is committed, and H&M's prices are scaled, so only relative depth will be meaningful there.
- Sell-out: sales are a sell-through proxy; stock levels and competitor prices are absent.
