"""Pure-pandas analytics used by the dashboard. Every function takes the
(already filtered) tweet DataFrame and returns a DataFrame or plain values."""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from . import config


# --------------------------------------------------------------------------------
# Campaign-level summary
# --------------------------------------------------------------------------------
def campaign_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per campaign with volume, sentiment and engagement metrics."""
    rows = []
    for campaign, g in df.groupby("campaign", sort=True):
        n = len(g)
        eng = g["engagement"].sort_values(ascending=False)
        top1_n = max(1, int(np.ceil(n * 0.01)))
        total_eng = int(g["engagement"].sum())
        pos = (g["vader_label"] == "Positive").mean() * 100
        neg = (g["vader_label"] == "Negative").mean() * 100
        rows.append(
            {
                "campaign": campaign,
                "tweets": n,
                "unique_hashtags": len({h for tags in g["hashtags"] for h in tags}),
                "avg_sentiment": g["vader_compound"].mean(),
                "positive_pct": pos,
                "neutral_pct": 100 - pos - neg,
                "negative_pct": neg,
                "net_sentiment": pos - neg,
                "total_likes": int(g["likes"].sum()),
                "total_retweets": int(g["retweet_count"].sum()),
                "total_engagement": total_eng,
                "avg_engagement": g["engagement"].mean(),
                "median_engagement": g["engagement"].median(),
                "engaged_tweets_pct": (g["engagement"] > 0).mean() * 100,
                "top1pct_engagement_share": (eng.head(top1_n).sum() / total_eng * 100) if total_eng else 0.0,
                "retweets_per_100_likes": (g["retweet_count"].sum() / g["likes"].sum() * 100) if g["likes"].sum() else 0.0,
            }
        )
    return pd.DataFrame(rows).round(3)


def daily_sentiment(df: pd.DataFrame, rolling_days: int = 3) -> pd.DataFrame:
    """Daily tweets, mean sentiment, net sentiment per campaign + rolling mean."""
    d = (
        df.assign(pos=df["vader_label"].eq("Positive"), neg=df["vader_label"].eq("Negative"))
        .groupby(["campaign", "date"])
        .agg(tweets=("text", "size"), avg_sentiment=("vader_compound", "mean"),
             pos=("pos", "mean"), neg=("neg", "mean"), engagement=("engagement", "sum"))
        .reset_index()
        .sort_values(["campaign", "date"])
    )
    d["net_sentiment"] = (d["pos"] - d["neg"]) * 100
    d["rolling_sentiment"] = d.groupby("campaign")["avg_sentiment"].transform(
        lambda s: s.rolling(rolling_days, min_periods=1).mean()
    )
    return d.drop(columns=["pos", "neg"])


def sentiment_distribution(df: pd.DataFrame, label_col: str = "vader_label") -> pd.DataFrame:
    """% of tweets per sentiment label per campaign (long format)."""
    ct = pd.crosstab(df["campaign"], pd.Categorical(df[label_col], config.SENTIMENT_ORDER), normalize="index") * 100
    out = ct.reset_index().melt(id_vars="campaign", var_name="sentiment", value_name="pct")
    out["sentiment"] = pd.Categorical(out["sentiment"], config.SENTIMENT_ORDER, ordered=True)
    return out.sort_values(["campaign", "sentiment"])


# --------------------------------------------------------------------------------
# Engagement
# --------------------------------------------------------------------------------
def engagement_by_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.groupby(["campaign", "vader_label"], observed=True)
        .agg(tweets=("text", "size"), avg_engagement=("engagement", "mean"),
             median_engagement=("engagement", "median"), avg_likes=("likes", "mean"),
             avg_retweets=("retweet_count", "mean"), engaged_pct=("engagement", lambda s: (s > 0).mean() * 100))
        .reset_index()
    )
    g["vader_label"] = pd.Categorical(g["vader_label"], config.SENTIMENT_ORDER, ordered=True)
    return g.sort_values(["campaign", "vader_label"]).round(3)


def hourly_activity(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["campaign", "hour"])
        .agg(tweets=("text", "size"), avg_engagement=("engagement", "mean"))
        .reset_index()
    )


def lorenz_curve(df: pd.DataFrame, points: int = 200) -> pd.DataFrame:
    """Cumulative share of engagement vs share of tweets (ordered low -> high).
    The further below the diagonal, the more engagement is concentrated in few tweets."""
    out = []
    for campaign, g in df.groupby("campaign", sort=True):
        e = np.sort(g["engagement"].to_numpy())
        total = e.sum()
        if total == 0 or len(e) < 2:
            continue
        cum = np.cumsum(e) / total
        x = np.arange(1, len(e) + 1) / len(e)
        idx = np.unique(np.linspace(0, len(e) - 1, points).astype(int))
        out.append(pd.DataFrame({"campaign": campaign, "tweet_share": x[idx] * 100, "engagement_share": cum[idx] * 100}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["campaign", "tweet_share", "engagement_share"])


def top_tweets(df: pd.DataFrame, by: str = "engagement", n: int = 10) -> pd.DataFrame:
    cols = ["created_at", "campaign", "tweet", "likes", "retweet_count", "engagement", "vader_compound", "vader_label"]
    return df.nlargest(n, by)[cols].reset_index(drop=True)


# --------------------------------------------------------------------------------
# Geography
# --------------------------------------------------------------------------------
def state_summary(df: pd.DataFrame, min_tweets: int = 30) -> pd.DataFrame:
    g = (
        df[df["state"] != ""]
        .groupby("state")
        .agg(tweets=("text", "size"), avg_sentiment=("vader_compound", "mean"),
             net_sentiment=("vader_label", lambda s: ((s == "Positive").mean() - (s == "Negative").mean()) * 100),
             avg_engagement=("engagement", "mean"))
        .reset_index()
    )
    g = g[g["tweets"] >= min_tweets].copy()
    g["code"] = g["state"].map(config.US_STATE_CODES)
    return g.sort_values("tweets", ascending=False).round(3)


# --------------------------------------------------------------------------------
# Hashtags
# --------------------------------------------------------------------------------
def explode_hashtags(df: pd.DataFrame, exclude: set[str] | None = None) -> pd.DataFrame:
    cols = ["campaign", "date", "hashtags", "vader_compound", "engagement"]
    ex = df[cols].explode("hashtags").dropna(subset=["hashtags"]).rename(columns={"hashtags": "hashtag"})
    if exclude:
        ex = ex[~ex["hashtag"].isin(exclude)]
    return ex


def top_hashtags(df: pd.DataFrame, n: int = 15, exclude: set[str] | None = None) -> pd.DataFrame:
    ex = explode_hashtags(df, exclude)
    g = (
        ex.groupby(["campaign", "hashtag"])
        .agg(tweets=("hashtag", "size"), avg_sentiment=("vader_compound", "mean"), engagement=("engagement", "sum"))
        .reset_index()
    )
    g = g.sort_values(["campaign", "tweets"], ascending=[True, False]).groupby("campaign").head(n)
    return g.round(3).reset_index(drop=True)


def hashtag_sentiment(df: pd.DataFrame, min_tweets: int = 20, exclude: set[str] | None = None) -> pd.DataFrame:
    ex = explode_hashtags(df, exclude)
    g = (
        ex.groupby("hashtag")
        .agg(tweets=("hashtag", "size"), avg_sentiment=("vader_compound", "mean"), engagement=("engagement", "sum"))
        .reset_index()
    )
    return g[g["tweets"] >= min_tweets].sort_values("avg_sentiment").round(3).reset_index(drop=True)


def trending_hashtags(
    df: pd.DataFrame, recent_days: int = 3, min_recent: int = 10, exclude: set[str] | None = None, top_n: int = 20
) -> tuple[pd.DataFrame, tuple[pd.Timestamp, pd.Timestamp]]:
    """Detect hashtags whose share of tweets jumped in the most recent window.

    recent window = last `recent_days` calendar days of the filtered data;
    baseline      = every earlier day.
    lift = (recent rate + eps) / (baseline rate + eps), rates are hashtag mentions per
    1,000 tweets in each window and eps is half a mention (Laplace-style smoothing so
    brand-new hashtags don't divide by zero). lift > 1 means the hashtag is accelerating.
    """
    empty = pd.DataFrame(columns=["hashtag", "recent_mentions", "baseline_mentions", "recent_per_1k",
                                  "baseline_per_1k", "lift", "status"])
    if df.empty:
        return empty, (pd.NaT, pd.NaT)
    last_day = df["date"].max()
    cutoff = last_day - timedelta(days=int(recent_days) - 1)
    recent, base = df[df["date"] >= cutoff], df[df["date"] < cutoff]
    window = (cutoff, last_day)
    if recent.empty or base.empty:
        return empty, window

    rc = explode_hashtags(recent, exclude)["hashtag"].value_counts().rename("recent_mentions")
    bc = explode_hashtags(base, exclude)["hashtag"].value_counts().rename("baseline_mentions")
    t = pd.concat([rc, bc], axis=1).fillna(0).astype(int)
    t = t[t["recent_mentions"] >= min_recent].copy()
    if t.empty:
        return empty, window
    t["recent_per_1k"] = t["recent_mentions"] / len(recent) * 1000
    t["baseline_per_1k"] = t["baseline_mentions"] / len(base) * 1000
    eps_recent, eps_base = 0.5 / len(recent) * 1000, 0.5 / len(base) * 1000
    t["lift"] = (t["recent_per_1k"] + eps_recent) / (t["baseline_per_1k"] + eps_base)
    t["status"] = np.where(t["baseline_mentions"] == 0, "New", np.where(t["lift"] >= 1.5, "Surging", np.where(t["lift"] <= 0.67, "Fading", "Stable")))
    t = t.reset_index(names="hashtag").sort_values(["lift", "recent_mentions"], ascending=False)
    return t.head(top_n).round(3).reset_index(drop=True), window


def hashtag_daily(df: pd.DataFrame, tags: list[str]) -> pd.DataFrame:
    ex = explode_hashtags(df)
    ex = ex[ex["hashtag"].isin(tags)]
    return ex.groupby(["date", "hashtag"]).size().rename("mentions").reset_index()


# --------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------
def bootstrap_mean_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 1000, seed: int = 42, max_n: int = 20_000):
    """Mean(a) - Mean(b) with a 95% bootstrap confidence interval."""
    rng = np.random.default_rng(seed)
    a = rng.choice(a, size=min(len(a), max_n), replace=False) if len(a) > max_n else a
    b = rng.choice(b, size=min(len(b), max_n), replace=False) if len(b) > max_n else b
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        diffs[i] = rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(a.mean() - b.mean()), float(lo), float(hi)


# --------------------------------------------------------------------------------
# Auto-generated insights (all derived from the filtered data)
# --------------------------------------------------------------------------------
def build_insights(summary: pd.DataFrame) -> list[str]:
    if summary.empty:
        return []
    s = summary.set_index("campaign")
    out = []
    big = s["tweets"].idxmax()
    out.append(f"**{big}** has the largest conversation: {int(s.loc[big, 'tweets']):,} tweets "
               f"({s.loc[big, 'tweets'] / s['tweets'].sum() * 100:.1f}% of the filtered data).")
    if len(s) > 1:
        best, worst = s["net_sentiment"].idxmax(), s["net_sentiment"].idxmin()
        out.append(f"**{best}** has the most favourable net sentiment ({s.loc[best, 'net_sentiment']:+.1f} pts "
                   f"positive minus negative) vs **{worst}** ({s.loc[worst, 'net_sentiment']:+.1f} pts).")
        eng = s["avg_engagement"].idxmax()
        out.append(f"**{eng}** earns more engagement per tweet ({s.loc[eng, 'avg_engagement']:.1f} "
                   f"likes + {config.RETWEET_WEIGHT}x retweets).")
    conc = s["top1pct_engagement_share"].idxmax()
    out.append(f"Engagement is highly concentrated: the top 1% of **{conc}** tweets capture "
               f"{s.loc[conc, 'top1pct_engagement_share']:.0f}% of that campaign's engagement.")
    return out


def cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    """Chance-corrected agreement between two labelings (1 = perfect, 0 = chance)."""
    po = float((a.to_numpy() == b.to_numpy()).mean())
    pa, pb = a.value_counts(normalize=True), b.value_counts(normalize=True)
    pe = float(sum(pa.get(k, 0) * pb.get(k, 0) for k in set(pa.index) | set(pb.index)))
    return (po - pe) / (1 - pe) if pe < 1 else 1.0
