"""Sentiment analysis: VADER (always available) and BERT-family models (optional)."""
from __future__ import annotations

import importlib.util
from typing import Callable

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from . import config


# --------------------------------------------------------------------------------
# VADER
# --------------------------------------------------------------------------------
def label_from_compound(compound: pd.Series) -> pd.Series:
    """Map VADER compound (-1..1) to Positive / Neutral / Negative."""
    thr = config.VADER_THRESHOLD
    return pd.Series(
        np.select([compound >= thr, compound <= -thr], ["Positive", "Negative"], default="Neutral"),
        index=compound.index,
    )


def add_vader(df: pd.DataFrame) -> pd.DataFrame:
    analyzer = SentimentIntensityAnalyzer()
    compound = [analyzer.polarity_scores(t)["compound"] for t in df["text"]]
    out = df.copy()
    out["vader_compound"] = np.asarray(compound, dtype="float64")
    out["vader_label"] = label_from_compound(out["vader_compound"])
    return out


# --------------------------------------------------------------------------------
# BERT-family (optional: needs `pip install -r requirements-bert.txt`)
# --------------------------------------------------------------------------------
def bert_available() -> bool:
    return all(importlib.util.find_spec(m) is not None for m in ("transformers", "torch"))


def normalize_label(label: str) -> str:
    """Map any model's label vocabulary onto Positive / Neutral / Negative."""
    low = str(label).strip().lower()
    if low in {"label_0"}:
        return "Negative"
    if low in {"label_1"}:
        return "Neutral"
    if low in {"label_2"}:
        return "Positive"
    if "star" in low:  # nlptown: "1 star" ... "5 stars"
        stars = int(low.split()[0])
        return "Negative" if stars <= 2 else "Neutral" if stars == 3 else "Positive"
    if low.startswith("neg"):
        return "Negative"
    if low.startswith("pos"):
        return "Positive"
    if low.startswith("neu"):
        return "Neutral"
    raise ValueError(f"Unrecognised sentiment label from model: {label!r}")


def load_bert_pipeline(model_name: str):
    """Download (first run only) and return a Hugging Face text-classification pipeline."""
    from transformers import pipeline  # imported lazily so the app runs without it

    return pipeline("text-classification", model=model_name, truncation=True, max_length=128)


def bert_predict(
    clf,
    texts: list[str],
    *,
    batch_size: int = 32,
    progress: Callable[[float], None] | None = None,
) -> pd.DataFrame:
    """Run the classifier in batches; returns columns bert_label, bert_score."""
    labels, scores = [], []
    total = max(len(texts), 1)
    for start in range(0, len(texts), batch_size):
        batch = [t if t.strip() else "." for t in texts[start : start + batch_size]]
        for res in clf(batch, batch_size=batch_size):
            labels.append(normalize_label(res["label"]))
            scores.append(float(res["score"]))
        if progress:
            progress(min(1.0, (start + batch_size) / total))
    return pd.DataFrame({"bert_label": labels, "bert_score": scores})


def agreement_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Row-normalised confusion matrix: VADER label (rows) vs BERT label (cols)."""
    ct = pd.crosstab(
        pd.Categorical(df["vader_label"], config.SENTIMENT_ORDER),
        pd.Categorical(df["bert_label"], config.SENTIMENT_ORDER),
        dropna=False,
    )
    ct.index.name, ct.columns.name = "VADER", "BERT"
    return ct
