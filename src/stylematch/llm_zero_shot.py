"""Zero-shot LLM attribute extraction (NOT run in the sandbox: no API key / no benchmark numbers yet).
Usage on your machine: ANTHROPIC_API_KEY=... python scripts/run_zero_shot.py  (needs `pip install anthropic`)."""
import json, os

PROMPT = """You label fashion products from a description only.
Return JSON with keys product_type (one of {types}), colour (one of {colours}), department (one of {depts}).
Description: {desc}"""


def build_prompt(desc, types, colours, depts):
    return PROMPT.format(types=types, colours=colours, depts=depts, desc=desc)


def parse(text):
    try: return json.loads(text[text.index("{"): text.rindex("}") + 1])
    except Exception: return {}


def predict(descs, types, colours, depts, model=None):
    import anthropic
    c = anthropic.Anthropic(); model = model or os.environ.get("ZS_MODEL", "claude-haiku-4-5")
    out = []
    for d in descs:
        r = c.messages.create(model=model, max_tokens=100, messages=[{"role": "user", "content": build_prompt(d, types, colours, depts)}])
        out.append(parse(r.content[0].text))
    return out
