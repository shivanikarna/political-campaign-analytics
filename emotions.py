"""Transparent, dependency-free emotion detection (Plutchik's 8 basic emotions).

Each emotion has a small hand-written lexicon below. A word ending in ``*`` matches
any word starting with that stem (``hate*`` -> hate, hated, hateful ...). This is a
dictionary approach: it is fast, explainable and fully reproducible, but it cannot
detect sarcasm or context. The lexicon is intentionally visible so it can be audited
or extended.
"""
from __future__ import annotations

import re

import pandas as pd

EMOTION_LEXICON: dict[str, list[str]] = {
    "Anger": [
        "angry", "anger*", "furious", "fury", "rage*", "enrag*", "outrag*", "hate*", "hating",
        "hatred", "mad", "pissed", "livid", "infuriat*", "fuck*", "damn*", "idiot*", "stupid",
        "moron*", "traitor*", "treason*", "thug*", "attack*", "fight*", "destroy*", "ruin*",
        "punish*", "disgrace*", "resent*",
    ],
    "Fear": [
        "afraid", "fear*", "scare*", "scary", "terrif*", "terror*", "panic*", "dread*", "threat*",
        "danger*", "worried", "worry", "worrying", "anxious", "anxiety", "nervous", "frighten*",
        "horrif*", "alarm*", "unsafe", "violence", "violent", "riot*", "chaos", "crisis", "collapse",
        "coup", "deadly", "catastroph*",
    ],
    "Disgust": [
        "disgust*", "gross", "sick", "sickening", "vile", "nasty", "revolting", "repuls*", "filth*",
        "trash", "garbage", "scum", "corrupt*", "liar*", "lie", "lies", "lied", "lying", "fraud*",
        "cheat*", "rigged", "crook*", "shameful", "despicable", "pathetic", "slimy", "toxic",
        "rotten", "sleaze*", "shame",
    ],
    "Sadness": [
        "sad", "sadly", "sadness", "sorrow*", "grief", "grieve*", "grieving", "cry", "cried",
        "crying", "tears", "depress*", "heartbreak*", "heartbroken", "devastat*", "mourn*",
        "tragic", "tragedy", "miserable", "hopeless", "despair*", "lonely", "loss", "lost",
        "disappoint*", "regret*", "unfortunate*", "suffer*", "pain", "painful", "hurt*", "death*",
        "dead", "died", "dying",
    ],
    "Joy": [
        "happy", "happiness", "joy*", "love*", "loved", "loving", "great", "wonderful", "amazing",
        "awesome", "excellent", "fantastic", "celebrat*", "win", "wins", "winning", "won", "winner*",
        "victory", "victorious", "proud", "pride", "delight*", "cheer*", "thrilled", "excited",
        "exciting", "congrat*", "blessed", "bless", "beautiful", "best", "glad", "smile*", "laugh*",
        "lol", "haha*", "fun", "yay", "hooray",
    ],
    "Trust": [
        "trust*", "honest*", "honesty", "loyal*", "faith*", "reliable", "integrity", "credible",
        "credibility", "support*", "believe*", "belief", "confidence", "confident", "protect*",
        "safe", "safety", "secure*", "united", "unity", "together", "leader*", "respect*", "dignity",
        "decency", "decent", "truth", "truthful", "fair", "fairness", "promise*", "dependable",
        "steady",
    ],
    "Surprise": [
        "surpris*", "shock*", "stunned", "stunning", "unexpected", "unbelievable", "incredible",
        "wow", "whoa", "omg", "amazed", "astonish*", "startl*", "sudden", "suddenly", "breaking",
        "unprecedented", "bombshell", "wtf",
    ],
    "Anticipation": [
        "soon", "wait", "waiting", "await*", "anticipat*", "expect*", "hope*", "hoping", "hopeful",
        "upcoming", "tomorrow", "tonight", "countdown", "ready", "prepar*", "predict*", "forecast*",
        "future",
    ],
}

EMOTIONS = list(EMOTION_LEXICON)


def _compile(words: list[str]) -> re.Pattern:
    parts = [re.escape(w[:-1]) + r"\w*" if w.endswith("*") else re.escape(w) for w in words]
    return re.compile(r"\b(?:" + "|".join(parts) + r")\b")


_PATTERNS = {emotion: _compile(words) for emotion, words in EMOTION_LEXICON.items()}


def add_emotions(df: pd.DataFrame) -> pd.DataFrame:
    """Add emo_<Emotion> hit counts, emotion_hits and dominant_emotion columns."""
    out = df.copy()
    lowered = out["text"].str.lower()
    for emotion, pattern in _PATTERNS.items():
        out[f"emo_{emotion}"] = lowered.str.count(pattern).astype("int32")
    cols = [f"emo_{e}" for e in EMOTIONS]
    out["emotion_hits"] = out[cols].sum(axis=1)
    dominant = out[cols].idxmax(axis=1).str.replace("emo_", "", regex=False)
    out["dominant_emotion"] = dominant.where(out["emotion_hits"] > 0, "None")
    return out


def emotion_share(df: pd.DataFrame, by: str = "campaign") -> pd.DataFrame:
    """% of tweets in each group that express each emotion (>=1 lexicon hit)."""
    present = (df[[f"emo_{e}" for e in EMOTIONS]] > 0).astype(float)
    present.columns = EMOTIONS
    share = present.groupby(df[by]).mean() * 100
    return share.round(2)
