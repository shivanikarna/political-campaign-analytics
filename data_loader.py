"""Load and clean the Kaggle "US Election 2020 Tweets" dataset.

Supported inputs (auto-detected, first match wins):
  1. data/*.csv or data/raw/*.csv  - the official Kaggle files
       hashtag_donaldtrump.csv, hashtag_joebiden.csv
     (campaign is taken from the file name), or any CSV that has a
     ``candidate`` column (pre-merged mirrors: trump_df / biden_df).
  2. data/sample/*.csv             - small real-data sample bundled with the repo.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

# Columns we keep from the (very wide) raw Kaggle files. Everything else is
# never read into memory.
WANTED_COLUMNS = {"created_at", "tweet", "likes", "retweet_count", "state", "country", "candidate"}

_URL = re.compile(r"https?://\S+|www\.\S+")
_MENTION = re.compile(r"@\w+")
_SPACES = re.compile(r"\s+")
_HASHTAG = r"#\w+"
_WORD = re.compile(r"[^\W\d_']+(?:'[^\W\d_]+)?")

# Very small function-word lists used by the English-only heuristic.
# Words that are also common English words (die, per, con, mit, la ...) are
# deliberately excluded so that English tweets are not misclassified.
_EN_WORDS = frozenset(
    "the and is are was were to of in that this it for with on you not be have has but they "
    "we he she his her our your at as so just what who why how all from by or if will would "
    "can do does did my me i am been their there about".split()
)
_FOREIGN_WORDS = frozenset(
    "el los las que por para una del les des pour dans une sont pas qui der das und nicht ist "
    "ein eine auf von zu sich auch il che sono gli nel della questo uma não são então você os "
    "se en y un es como más pero muy está".split()
)


# --------------------------------------------------------------------------------
# File discovery
# --------------------------------------------------------------------------------
def find_data_files() -> tuple[list[Path], str]:
    """Return (files, human-readable source label)."""
    full = sorted(config.DATA_DIR.glob("*.csv")) + sorted((config.DATA_DIR / "raw").glob("*.csv"))
    if full:
        return full, "Kaggle dataset (data/)"
    sample = sorted(config.SAMPLE_DIR.glob("*.csv"))
    if sample:
        return sample, "Bundled real-data sample (data/sample/)"
    return [], "none"


def file_signature(files: list[Path]) -> tuple:
    """Hashable signature so Streamlit's cache refreshes when files change."""
    return tuple((str(f), f.stat().st_size, f.stat().st_mtime_ns) for f in files)


# --------------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------------
def _campaign_label(value: str) -> str:
    low = str(value).strip().lower()
    for keyword, name in config.CAMPAIGN_KEYWORDS.items():
        if keyword in low:
            return name
    return re.sub(r"_df$", "", low).strip().title() or "Unknown"


def read_file(path: Path) -> pd.DataFrame:
    """Read one CSV, keeping only the columns we need."""
    df = pd.read_csv(
        path,
        # The Kaggle files contain line breaks inside quoted tweets and use
        # Windows line endings; '\n' as the record terminator parses them reliably.
        lineterminator="\n",
        usecols=lambda c: c.strip() in WANTED_COLUMNS,
        low_memory=False,
        on_bad_lines="skip",
        encoding_errors="replace",
    )
    df.columns = [c.strip() for c in df.columns]
    if "tweet" not in df or "created_at" not in df:
        raise ValueError(f"{path.name}: expected at least 'created_at' and 'tweet' columns, found {list(df.columns)}")

    if "candidate" in df:
        df["campaign"] = df["candidate"].astype(str).map(_campaign_label)
        df = df.drop(columns="candidate")
    else:
        df["campaign"] = _campaign_label(path.stem)

    for col in ("state", "country"):
        if col not in df:
            df[col] = ""
    for col in ("likes", "retweet_count"):
        if col not in df:
            df[col] = 0
    return df


# --------------------------------------------------------------------------------
# Cleaning helpers
# --------------------------------------------------------------------------------
def clean_text(s: pd.Series) -> pd.Series:
    """Text used for sentiment/emotion models: no URLs, @mentions or '#' signs."""
    s = s.str.replace("&amp;", "&", regex=False).str.replace("&lt;", "<", regex=False)
    s = s.str.replace("&gt;", ">", regex=False).str.replace("&quot;", '"', regex=False)
    s = s.str.replace("&#39;", "'", regex=False)
    s = s.str.replace(_URL, " ", regex=True).str.replace(_MENTION, " ", regex=True)
    s = s.str.replace("#", "", regex=False)
    return s.str.replace(_SPACES, " ", regex=True).str.strip()


def extract_hashtags(s: pd.Series) -> pd.Series:
    """Unique, lower-cased hashtags per tweet (a tweet repeating #x counts once)."""
    return s.str.findall(_HASHTAG).map(lambda tags: sorted({t.lower() for t in tags}))


def looks_english(text: str) -> bool:
    """Cheap, dependency-free English detector (heuristic, documented in the UI)."""
    letters = re.findall(r"[^\W\d_]", text)
    if not letters:
        return True  # emoji / numbers only: keep (VADER can score emoji)
    if sum(ch.isascii() for ch in letters) / len(letters) < 0.85:
        return False
    tokens = [t.lower() for t in _WORD.findall(text)]
    english = sum(t in _EN_WORDS for t in tokens)
    foreign = sum(t in _FOREIGN_WORDS for t in tokens)
    return english >= foreign


# --------------------------------------------------------------------------------
# Full pipeline
# --------------------------------------------------------------------------------
def load_clean(
    files: list[Path],
    *,
    max_rows_per_campaign: int | None,
    english_only: bool = True,
    dedupe: bool = True,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Read, validate, de-duplicate, sample and enrich the tweets.

    Returns (dataframe, report) where report lists how many rows each step removed.
    """
    if not files:
        raise FileNotFoundError("No CSV files found in data/ or data/sample/.")

    df = pd.concat([read_file(f) for f in files], ignore_index=True)
    steps = [("Rows loaded", len(df), 0)]

    def record(label: str, before: int) -> None:
        steps.append((label, len(df), before - len(df)))

    # 1) types & basic validity ---------------------------------------------------
    n = len(df)
    df["tweet"] = df["tweet"].astype("string").str.strip()
    df["created_at"] = pd.to_datetime(df["created_at"].astype("string").str.strip(), errors="coerce")
    df = df[df["tweet"].notna() & (df["tweet"] != "") & df["created_at"].notna()].copy()
    record("Valid text + timestamp", n)

    for col in ("likes", "retweet_count"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0).astype("int64")
    for col in ("state", "country"):
        df[col] = df[col].astype("string").str.strip().fillna("")
    df["tweet"] = df["tweet"].astype(str)
    df["engagement"] = df["likes"] + config.RETWEET_WEIGHT * df["retweet_count"]

    # 2) exact-duplicate removal (keep the copy with the most engagement) --------
    if dedupe:
        n = len(df)
        df["_key"] = df["tweet"].str.lower().str.strip()
        df = df.sort_values("engagement", ascending=False, kind="stable")
        df = df.drop_duplicates(subset=["campaign", "_key"], keep="first").drop(columns="_key")
        record("Duplicate tweets removed", n)

    # 3) sampling cap per campaign ------------------------------------------------
    if max_rows_per_campaign:
        n = len(df)
        parts = [
            g.sample(n=min(len(g), max_rows_per_campaign), random_state=seed)
            for _, g in df.groupby("campaign", sort=True)
        ]
        df = pd.concat(parts)
        record(f"Random sample (<= {max_rows_per_campaign:,} per campaign)", n)

    # 4) text enrichment ----------------------------------------------------------
    df["text"] = clean_text(df["tweet"])
    df["hashtags"] = extract_hashtags(df["tweet"])
    n = len(df)
    df = df[df["text"] != ""].copy()  # tweets that were only URLs / @mentions
    record("Empty after cleaning removed", n)

    if english_only:
        n = len(df)
        df = df[df["text"].map(looks_english)].copy()
        record("Non-English removed (heuristic)", n)

    # 5) final columns --------------------------------------------------------------
    df = df.sort_values("created_at").reset_index(drop=True)
    df["date"] = df["created_at"].dt.normalize()
    df["hour"] = df["created_at"].dt.hour
    df["n_hashtags"] = df["hashtags"].map(len)

    report = {
        "files": [f.name for f in files],
        "steps": steps,
        "rows_final": len(df),
        "campaigns": sorted(df["campaign"].unique().tolist()),
        "date_min": df["created_at"].min(),
        "date_max": df["created_at"].max(),
    }
    return df, report


def campaign_colors(campaigns: list[str]) -> dict[str, str]:
    """Stable colour per campaign (red/blue for Trump/Biden, palette for others)."""
    colors, extra = {}, iter(config.FALLBACK_COLORS)
    for c in campaigns:
        colors[c] = config.CAMPAIGN_COLORS.get(c) or next(extra, "#64748B")
    return colors
