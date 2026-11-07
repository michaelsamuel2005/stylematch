import json
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
m = json.load(open("reports/match_tfidf-svd.json")); a = json.load(open("reports/attributes.json")); p = json.load(open("reports/price.json"))
INK, MUTE, ACC = RGBColor(0x1b,0x1f,0x2a), RGBColor(0x5a,0x61,0x70), RGBColor(0x1f,0x6f,0xeb)
prs = Presentation(); prs.slide_width, prs.slide_height = Inches(13.33), Inches(7.5)
def slide(title, bullets=None, table=None, foot="Simulated data (stylematch twin). Not real retailer data."):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    t = s.shapes.add_textbox(Inches(.6), Inches(.4), Inches(12.1), Inches(1.3)).text_frame; t.word_wrap = True
    t.text = title; r = t.paragraphs[0].runs[0]; r.font.size = Pt(30); r.font.bold = True; r.font.color.rgb = INK
    y = 1.9
    if bullets:
        tf = s.shapes.add_textbox(Inches(.6), Inches(y), Inches(12.1 if not table else 5.6), Inches(4.5)).text_frame; tf.word_wrap = True
        for i, b in enumerate(bullets):
            para = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); para.text = b; para.space_after = Pt(14)
            para.runs[0].font.size = Pt(19); para.runs[0].font.color.rgb = INK
    if table:
        rows, cols = len(table), len(table[0]); x = 6.5 if bullets else .6
        g = s.shapes.add_table(rows, cols, Inches(x), Inches(y), Inches(12.7 - x), Inches(.45 * rows)).table
        for i, row in enumerate(table):
            for j, v in enumerate(row):
                c = g.cell(i, j); c.text = str(v); c.text_frame.paragraphs[0].runs[0].font.size = Pt(14)
    f = s.shapes.add_textbox(Inches(.6), Inches(6.85), Inches(12), Inches(.4)).text_frame; f.text = foot
    f.paragraphs[0].runs[0].font.size = Pt(11); f.paragraphs[0].runs[0].font.color.rgb = MUTE
t, ti = m["text_only"], m["text_plus_image_SIMULATED_w0.25"]
slide("What a buyer can learn from product, label and price data, and where to be careful",
      ["Same product, different colourway or department: found automatically.",
       "Wrong labels: caught for review, but not safe to auto-correct.",
       "Price architecture and markdown timing: reconstructed from sales alone.",
       "Everything here is a simulated catalogue with a known answer key. It shows the method works, not what any retailer's market looks like."])
slide("Matching: finding the same style across colourways and departments",
      [f"Text-only matching puts the true match first for {t['p_at_1']:.0%} of products and finds {t['recall_at_k']:.0%} of all true matches in the top 10.",
       f"Adding a simulated image signal moved ranking quality from {t['ndcg']:.3f} to {ti['ndcg']:.3f}: too small to claim a gain from one run.",
       "Real image embeddings (CLIP) have not been tested yet."],
      [["Method", "NDCG@10", "Match first"], ["Text only", f"{t['ndcg']:.3f}", f"{t['p_at_1']:.0%}"], ["Text + image (simulated)", f"{ti['ndcg']:.3f}", f"{ti['p_at_1']:.0%}"]])
acc = {r["attribute"]: r for r in a["scores"]}; ed = {r["attribute"]: r for r in a["error_detection"]}
slide("Labels: colour and type are reliable, department is not",
      ["Department is the weak spot: many descriptions never say who the item is for, so men's and women's items get swapped.",
       f"Flagging disagreeing labels catches {ed['department']['recall']:.0%} of planted department errors but only {ed['department']['precision']:.0%} of flags are real errors. Use it as a review list.",
       "Product type is easy in this simulation; do not read 99.8% as a real-world benchmark."],
      [["Attribute", "Accuracy"]] + [[k.replace("_", " ").title(), f"{acc[k]['accuracy']:.1%}"] for k in ("product_type", "colour", "department")])
arch = p["price_architecture"]
slide("Price architecture: where each category sits and how often it is marked down",
      ["First prices are reconstructed from sales, not read from a price file.", "In this simulation categories look almost the same by construction (similar prices and markdown rates), so there is no real category story here. The view is ready for real data."],
      [["Category", "Median price", "Marked down", "Median depth"]] + [[r["product_type"], f"{r['median']:.0f}", f"{r['markdown_rate']:.0%}", f"{r['median_depth']:.0%}"] for r in arch])
d, mm = p["markdown_detection_vs_planted"], p["markdown_timing_model"]
slide("Markdowns: detectable from sales, and slow starters go first",
      [f"Markdown detection: no false alarms and {d['recall']:.0%} of planted markdowns found; timing off by {d['offset_mae_weeks']:.1f} weeks on average.",
       f"Sales in the first four weeks predict when a markdown arrives: error {mm['mae_gbm']:.1f} weeks against {mm['mae_type_mean']:.1f} using the category average.",
       "Caveat: this relationship was planted in the simulation. On real data it must be re-tested."])
f = p["forecast_wape_backtest"]
slide("What this data cannot support",
      [f"Forecasts: the simple recent-average beat seasonal models (error {f['naive_last4']:.0%} vs {f['seasonal_naive']:.0%}) because products launch and fade within the window. Two years is too short to learn seasons.",
       "No competitor prices, stock levels or web-crawled listings, so sell-through is a proxy and not sell-out.",
       "No causal price-elasticity claims: the simulation's price response is an assumption.",
       "Next step: run on real catalogue data with real image embeddings and compare against these baselines."])
prs.save("deck/stylematch_buyer_deck.pptx"); print("saved")
