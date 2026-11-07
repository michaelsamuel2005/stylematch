"""Part 4: serve the matcher. POST /match {description, k} -> nearest catalogue articles."""
import os
import numpy as np, pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from stylematch.match import TfidfEncoder

STATE = {}


class Query(BaseModel):
    description: str = Field(min_length=3, max_length=1000)
    k: int = Field(5, ge=1, le=50)


def load(path=None):
    path = path or os.environ.get("STYLEMATCH_ARTICLES", "data/articles.csv")
    art = pd.read_csv(path)
    enc = TfidfEncoder(); E = enc.fit_transform(art.description)
    STATE.update(art=art, enc=enc, E=E)


@asynccontextmanager
async def lifespan(_):
    if os.path.exists(os.environ.get("STYLEMATCH_ARTICLES", "data/articles.csv")): load()
    yield


app = FastAPI(title="stylematch", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health(): return {"status": "ok", "articles": int(len(STATE["art"])) if "art" in STATE else 0}


@app.post("/match")
def match(q: Query):
    if "art" not in STATE: raise HTTPException(503, "index not loaded")
    s = STATE["E"] @ STATE["enc"].transform([q.description])[0]
    top = np.argsort(-s)[:q.k]; a = STATE["art"]
    return {"matches": [dict(article_id=int(a.article_id.iloc[i]), score=round(float(s[i]), 4), description=a.description.iloc[i],
                             colour=a.colour.iloc[i], department=a.department.iloc[i]) for i in top]}
