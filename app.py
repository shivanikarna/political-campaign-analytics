"""Political Campaign Analytics Platform  -  run with:  streamlit run app.py"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import analytics as an
from src import config
from src import data_loader as dl
from src import emotions as em
from src import sentiment as sm
from src import ui

st.set_page_config(page_title=config.APP_TITLE, page_icon="🗳️", layout="wide", initial_sidebar_state="expanded")
ui.inject_css()

FILTER_KEYS = ["f_campaigns", "f_dates", "f_sent", "f_states", "f_mineng", "f_kw", "f_tag"]


# ------------------------------------------------------------------------------------
# Cached heavy work
# ------------------------------------------------------------------------------------
@st.cache_data(show_spinner="Cleaning data and scoring sentiment + emotions (cached after the first run)...")
def build_dataset(signature: tuple, cap: int | None, english_only: bool, dedupe: bool):
    files = [Path(p) for p, _, _ in signature]
    df, report = dl.load_clean(files, max_rows_per_campaign=cap, english_only=english_only, dedupe=dedupe)
    df = sm.add_vader(df)
    df = em.add_emotions(df)
    return df, report


@st.cache_resource(show_spinner="Loading BERT model (the first run downloads it)...")
def get_bert(model_name: str):
    return sm.load_bert_pipeline(model_name)


@st.cache_data(show_spinner=False)
def bootstrap(a: np.ndarray, b: np.ndarray):
    return an.bootstrap_mean_diff(a, b, n_boot=500)


def to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")  # BOM so Excel on Windows shows emoji/accents correctly


def reset_filters() -> None:
    for key in FILTER_KEYS:
        st.session_state.pop(key, None)


# ------------------------------------------------------------------------------------
# Setup screen (only when no data is found)
# ------------------------------------------------------------------------------------
def setup_screen() -> None:
    ui.hero(config.APP_TITLE, "No dataset found yet", ["setup required"])
    st.error("No CSV files were found in `data/` or `data/sample/`.")
    st.markdown(
        f"""
**Get the real dataset (free Kaggle account needed):**
1. Open <{config.KAGGLE_URL}> and click **Download**.
2. Unzip and copy `hashtag_donaldtrump.csv` and `hashtag_joebiden.csv` into the project's `data/` folder.
3. Restart the app.

Or from PowerShell: `pip install kaggle` then `python scripts/download_data.py` (see README).
"""
    )


# ------------------------------------------------------------------------------------
# Tabs
# ------------------------------------------------------------------------------------
def tab_overview(fdf, summary, colors):
    ui.section("Key performance indicators", "Every card recalculates from the filters in the sidebar.")
    view = st.radio("KPI view", ["Combined", "Per campaign"], horizontal=True, key="kpi_view", label_visibility="collapsed")
    if view == "Combined":
        pos = (fdf["vader_label"] == "Positive").mean() * 100
        neg = (fdf["vader_label"] == "Negative").mean() * 100
        top = an.explode_hashtags(fdf, config.CORE_HASHTAGS)["hashtag"].value_counts().head(1)
        c = st.columns(6)
        c[0].metric("Tweets", f"{len(fdf):,}")
        c[1].metric("Avg sentiment", f"{fdf['vader_compound'].mean():+.3f}")
        c[2].metric("Net sentiment", f"{pos - neg:+.1f} pts", help="% positive minus % negative tweets")
        c[3].metric("Engagement", f"{int(fdf['engagement'].sum()):,}",
                    help=f"Total engagement score = likes + {config.RETWEET_WEIGHT} x retweets")
        c[4].metric("Avg engagement", f"{fdf['engagement'].mean():.2f}", help="Engagement score per tweet")
        c[5].metric("Top hashtag", top.index[0] if len(top) else "-",
                    help=f"{int(top.iloc[0]):,} tweets (campaign-name hashtags excluded)" if len(top) else None)
    else:
        for _, row in summary.iterrows():
            others = summary[summary["campaign"] != row["campaign"]]

            def delta(col, fmt):
                if others.empty:
                    return None
                return f"{row[col] - others[col].mean():{fmt}} vs others"

            st.markdown(f"**{row['campaign']}**")
            c = st.columns(5)
            c[0].metric("Tweets", f"{int(row['tweets']):,}", delta("tweets", "+,.0f"))
            c[1].metric("Avg sentiment", f"{row['avg_sentiment']:+.3f}", delta("avg_sentiment", "+.3f"))
            c[2].metric("Net sentiment", f"{row['net_sentiment']:+.1f} pts", delta("net_sentiment", "+.1f"))
            c[3].metric("Avg engagement", f"{row['avg_engagement']:.2f}", delta("avg_engagement", "+.2f"))
            c[4].metric("Total engagement", f"{int(row['total_engagement']):,}", delta("total_engagement", "+,.0f"))

    ui.section("What the data says")
    for line in an.build_insights(summary):
        ui.insight(line)

    daily = an.daily_sentiment(fdf)
    c1, c2 = st.columns([3, 2])
    with c1:
        fig = px.line(daily, x="date", y="tweets", color="campaign", markers=True,
                      color_discrete_map=colors, title="Tweet volume per day")
        ui.show(fig, 360)
    with c2:
        share = fdf["campaign"].value_counts().rename_axis("campaign").reset_index(name="tweets")
        fig = px.pie(share, names="campaign", values="tweets", hole=0.55, color="campaign",
                     color_discrete_map=colors, title="Share of conversation")
        ui.show(fig, 360, legend="right")
    fig = px.line(daily, x="date", y="rolling_sentiment", color="campaign", markers=True,
                  color_discrete_map=colors, title="Average VADER sentiment (3-day rolling mean)")
    fig.add_hline(y=0, line_dash="dot", line_color="#94A3B8")
    ui.show(fig, 340)


def tab_sentiment(fdf, colors, data_key):
    ui.section("VADER sentiment", f"Compound score from -1 (very negative) to +1 (very positive); "
                                  f"|score| < {config.VADER_THRESHOLD} counts as neutral.")
    dist = an.sentiment_distribution(fdf)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(dist, x="campaign", y="pct", color="sentiment", color_discrete_map=config.SENTIMENT_COLORS,
                     category_orders={"sentiment": config.SENTIMENT_ORDER}, text=dist["pct"].round(1).astype(str) + "%",
                     title="Sentiment mix by campaign", labels={"pct": "% of tweets"})
        fig.update_layout(barmode="stack")
        ui.show(fig, 380)
    with c2:
        fig = px.histogram(fdf, x="vader_compound", color="campaign", nbins=41, barmode="overlay", opacity=0.6,
                           histnorm="percent", color_discrete_map=colors, title="Distribution of VADER scores",
                           labels={"vader_compound": "VADER compound", "percent": "% of tweets"})
        ui.show(fig, 380)

    daily = an.daily_sentiment(fdf)
    fig = px.bar(daily, x="date", y="net_sentiment", color="campaign", barmode="group", color_discrete_map=colors,
                 title="Daily net sentiment (% positive - % negative)")
    fig.add_hline(y=0, line_color="#94A3B8")
    ui.show(fig, 340)

    # ---------------- VADER vs BERT ----------------
    ui.section("VADER vs BERT", "Lexicon-based VADER compared with a pretrained transformer on the same tweets.")
    if not sm.bert_available():
        st.info(
            "BERT is **optional** and not installed, so this section is disabled - everything else works without it.\n\n"
            "To enable it (PowerShell, inside your virtual environment):\n\n"
            "`pip install -r requirements-bert.txt`  then restart the app. "
            "The model (~260-670 MB) downloads automatically on the first run."
        )
        return

    c1, c2, c3 = st.columns([3, 2, 1])
    model = c1.selectbox("Model", list(config.BERT_MODELS), format_func=lambda m: f"{m}  -  {config.BERT_MODELS[m]}")
    max_n = min(5000, len(fdf))
    if max_n < 20:
        c2.warning("Too few tweets after filtering.")
        n = max_n
    elif max_n <= 50:
        n = max_n
        c2.caption(f"Scoring all {n} tweets")
    else:
        n = c2.slider("Tweets to score (stratified by campaign)", 50, max_n, min(1000, max_n), step=50)
    c3.write("")
    run = c3.button("Run BERT", type="primary", disabled=max_n < 20)

    if run:
        per = max(1, n // fdf["campaign"].nunique())
        sample = pd.concat([g.sample(min(len(g), per), random_state=42) for _, g in fdf.groupby("campaign")])
        try:
            clf = get_bert(model)
            bar = st.progress(0.0, text="Scoring tweets with BERT...")
            res = sm.bert_predict(clf, sample["text"].tolist(),
                                  progress=lambda p: bar.progress(p, text=f"Scoring tweets with BERT... {p:.0%}"))
            bar.empty()
            res.index = sample.index
            st.session_state["bert"] = {"key": (data_key, model), "df": res}
        except Exception as exc:  # e.g. offline first run
            st.error(f"BERT failed: {exc}\n\nThe first run needs internet access to download the model.")

    state = st.session_state.get("bert")
    if not state or state["key"] != (data_key, model):
        st.caption("Press **Run BERT** to score a sample and compare it with VADER.")
        return
    res = state["df"].join(fdf[["campaign", "vader_label", "vader_compound", "tweet"]], how="inner")
    if res.empty:
        st.info("None of the BERT-scored tweets match the current filters - press Run BERT again.")
        return
    reports["bert_vs_vader"] = res.reset_index(drop=True)

    agree = (res["vader_label"] == res["bert_label"]).mean() * 100
    kappa = an.cohen_kappa(res["vader_label"], res["bert_label"])
    m = st.columns(3)
    m[0].metric("Tweets compared", f"{len(res):,}")
    m[1].metric("Label agreement", f"{agree:.1f}%")
    m[2].metric("Cohen's kappa", f"{kappa:.2f}", help="Agreement beyond chance. <0.2 slight, 0.4-0.6 moderate, >0.6 substantial.")

    long = pd.concat([
        res.assign(model="VADER", label=res["vader_label"]),
        res.assign(model="BERT", label=res["bert_label"]),
    ])
    mix = (long.groupby(["campaign", "model"])["label"].value_counts(normalize=True).mul(100)
           .rename("pct").reset_index())
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(mix, x="model", y="pct", color="label", facet_col="campaign", color_discrete_map=config.SENTIMENT_COLORS,
                     category_orders={"label": config.SENTIMENT_ORDER, "model": ["VADER", "BERT"]},
                     title="Sentiment mix: VADER vs BERT", labels={"pct": "% of tweets"})
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        ui.show(fig, 380)
    with c2:
        cm = sm.agreement_matrix(res)
        pct = cm.div(cm.sum(axis=1).replace(0, np.nan), axis=0) * 100
        fig = px.imshow(pct.round(1), text_auto=".1f", color_continuous_scale="Blues", aspect="auto",
                        title="Where they agree: % of each VADER label assigned to each BERT label",
                        labels=dict(x="BERT says", y="VADER says", color="%"))
        ui.show(fig, 380)

    net = long.groupby(["campaign", "model"])["label"].agg(
        lambda s: ((s == "Positive").mean() - (s == "Negative").mean()) * 100).rename("net").reset_index()
    fig = px.bar(net, x="campaign", y="net", color="model", barmode="group",
                 color_discrete_map={"VADER": "#6366F1", "BERT": "#F59E0B"},
                 title="Net sentiment by campaign: does the ranking change with the model?",
                 labels={"net": "% positive - % negative"})
    ui.show(fig, 340)

    ui.section("Where the models disagree", "Highest-confidence BERT predictions that contradict VADER.")
    dis = res[res["vader_label"] != res["bert_label"]].nlargest(15, "bert_score")
    ui.table(dis[["campaign", "tweet", "vader_label", "vader_compound", "bert_label", "bert_score"]].round(3))


def tab_comparison(fdf, summary, colors):
    ui.section("Campaign comparison", "Side-by-side metrics for the campaigns selected in the sidebar.")
    if len(summary) < 2:
        st.info("Select at least two campaigns in the sidebar to compare them.")
    metrics = {
        "Tweets": "tweets", "Average sentiment (VADER)": "avg_sentiment", "Net sentiment (pos% - neg%)": "net_sentiment",
        "Positive %": "positive_pct", "Negative %": "negative_pct", "Avg engagement / tweet": "avg_engagement",
        "Total engagement": "total_engagement", "Engaged tweets %": "engaged_tweets_pct",
        "Top-1% tweets' engagement share": "top1pct_engagement_share", "Retweets per 100 likes": "retweets_per_100_likes",
        "Unique hashtags": "unique_hashtags",
    }
    c1, c2 = st.columns([3, 2])
    with c1:
        label = st.selectbox("Metric to compare", list(metrics), key="cmp_metric")
        col = metrics[label]
        fig = px.bar(summary, x="campaign", y=col, color="campaign", color_discrete_map=colors,
                     text=summary[col].map(lambda v: f"{v:,.2f}" if abs(v) < 1000 else f"{v:,.0f}"), title=label)
        fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title=None)
        ui.show(fig, 380)
    with c2:
        radar_cols = {"Volume": "tweets", "Sentiment": "avg_sentiment", "Positive %": "positive_pct",
                      "Engagement": "avg_engagement", "Engaged tweets %": "engaged_tweets_pct",
                      "Hashtag variety": "unique_hashtags"}
        fig = go.Figure()
        s = summary.set_index("campaign")
        norm = pd.DataFrame({k: s[v] for k, v in radar_cols.items()})
        norm["Sentiment"] = (norm["Sentiment"] + 1) / 2  # shift -1..1 to 0..1 so ratios make sense
        norm = norm / norm.max().replace(0, np.nan) * 100
        for campaign in norm.index:
            vals = norm.loc[campaign].fillna(0).tolist()
            fig.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=list(norm.columns) + [list(norm.columns)[0]],
                                          fill="toself", name=campaign, line_color=colors[campaign], opacity=0.55))
        fig.update_layout(title="Profile (100 = leading campaign on that axis)",
                          polar=dict(domain=dict(x=[0.18, 0.82]), radialaxis=dict(range=[0, 100], showticklabels=False)))
        ui.show(fig, 380)

    if len(summary) == 2:
        a_name, b_name = summary["campaign"].tolist()
        a = fdf.loc[fdf["campaign"] == a_name, "vader_compound"].to_numpy()
        b = fdf.loc[fdf["campaign"] == b_name, "vader_compound"].to_numpy()
        diff, lo, hi = bootstrap(a, b)
        sig = "does **not** include 0, so the gap is statistically meaningful" if (lo > 0 or hi < 0) else "includes 0, so the gap could be noise"
        ui.insight(f"Mean VADER sentiment difference **{a_name} - {b_name} = {diff:+.3f}** "
                   f"(95% bootstrap CI {lo:+.3f} to {hi:+.3f}); the interval {sig}.")

    c1, c2 = st.columns(2)
    with c1:
        share = em.emotion_share(fdf).reset_index().melt(id_vars="campaign", var_name="emotion", value_name="pct")
        fig = px.bar(share, x="emotion", y="pct", color="campaign", barmode="group", color_discrete_map=colors,
                     title="Emotion prevalence (% of tweets)", labels={"pct": "% of tweets"})
        ui.show(fig, 360)
    with c2:
        dist = an.sentiment_distribution(fdf)
        fig = px.bar(dist, x="pct", y="campaign", color="sentiment", orientation="h", color_discrete_map=config.SENTIMENT_COLORS,
                     category_orders={"sentiment": config.SENTIMENT_ORDER}, title="Sentiment mix", labels={"pct": "% of tweets"})
        fig.update_layout(barmode="stack")
        ui.show(fig, 360)

    ui.section("Full metric table")
    pretty = {"tweets": "Tweets", "unique_hashtags": "Unique hashtags", "avg_sentiment": "Avg sentiment",
              "positive_pct": "Positive %", "neutral_pct": "Neutral %", "negative_pct": "Negative %",
              "net_sentiment": "Net sentiment (pts)", "total_likes": "Total likes", "total_retweets": "Total retweets",
              "total_engagement": "Total engagement", "avg_engagement": "Avg engagement / tweet",
              "median_engagement": "Median engagement", "engaged_tweets_pct": "Engaged tweets %",
              "top1pct_engagement_share": "Top-1% engagement share %", "retweets_per_100_likes": "Retweets per 100 likes"}
    t = summary.set_index("campaign")[list(pretty)].rename(columns=pretty).T
    fmt = t.copy().astype(object)
    for metric_name in t.index:
        for camp in t.columns:
            v = t.loc[metric_name, camp]
            fmt.loc[metric_name, camp] = (f"{v:+.3f}" if metric_name == "Avg sentiment"
                                          else f"{v:,.0f}" if float(v).is_integer() and abs(v) >= 100 else f"{v:,.2f}")
    fmt.index.name = "Metric"
    ui.table(fmt.reset_index(), height=35 * (len(fmt) + 1) + 3)


def tab_hashtags(fdf, colors):
    ui.section("Hashtag intelligence", "Top hashtags, momentum-based trend detection and the sentiment each hashtag carries.")
    c = st.columns(3)
    hide_core = c[0].checkbox("Hide campaign-name hashtags (#trump, #biden ...)", True, key="h_core")
    top_n = c[1].slider("Hashtags per campaign", 5, 25, 12, key="h_topn")
    exclude = config.CORE_HASHTAGS if hide_core else set()

    top = an.top_hashtags(fdf, top_n, exclude)
    reports["top_hashtags"] = top
    if top.empty:
        st.info("No hashtags in the current selection.")
        return
    fig = px.bar(top, x="tweets", y="hashtag", color="avg_sentiment", facet_col="campaign", orientation="h",
                 color_continuous_scale="RdYlGn", range_color=[-0.5, 0.5], facet_col_spacing=0.2,
                 title="Most used hashtags (bar colour = average VADER sentiment)", labels={"avg_sentiment": "Avg sentiment"})
    fig.update_yaxes(matches=None, showticklabels=True, categoryorder="total ascending", title=None)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    ui.show(fig, 120 + 24 * top_n)

    ui.section("Trending hashtag detection",
               "Compares each hashtag's rate per 1,000 tweets in the most recent days against all earlier days.")
    n_days = fdf["date"].nunique()
    if n_days < 2:
        st.info("Trend detection needs at least 2 distinct days - widen the date filter.")
    else:
        c = st.columns(3)
        scope = c[0].selectbox("Campaign scope", ["All selected"] + sorted(fdf["campaign"].unique()), key="t_scope")
        recent_days = c[1].slider("Recent window (days)", 1, min(10, n_days - 1), min(3, n_days - 1), key="t_days")
        min_recent = c[2].slider("Min mentions in window", 3, 100, 10, key="t_min")
        sub = fdf if scope == "All selected" else fdf[fdf["campaign"] == scope]
        trend, (w0, w1) = an.trending_hashtags(sub, recent_days, min_recent, exclude)
        if trend.empty:
            st.info("No hashtag met the minimum-mentions threshold in the recent window. Lower the threshold.")
        else:
            reports["trending_hashtags"] = trend.assign(window_start=w0.date(), window_end=w1.date(), scope=scope)
            st.caption(f"Recent window: **{w0:%b %d} - {w1:%b %d, %Y}**  |  baseline: all earlier days in the selection")
            status_colors = {"New": "#7C3AED", "Surging": "#DC2626", "Fading": "#64748B", "Stable": "#94A3B8"}
            c1, c2 = st.columns([1, 1])
            with c1:
                fig = px.bar(trend.sort_values("lift"), x="lift", y="hashtag", color="status", orientation="h",
                             color_discrete_map=status_colors, title="Momentum (lift = recent rate / baseline rate)",
                             hover_data=["recent_mentions", "baseline_mentions"])
                fig.add_vline(x=1, line_dash="dot", line_color="#0F172A")
                ui.show(fig, 120 + 24 * len(trend))
            with c2:
                ui.table(trend[["hashtag", "status", "lift", "recent_mentions", "baseline_mentions"]],
                         column_config={"lift": st.column_config.NumberColumn(format="%.2f")})
            daily = an.hashtag_daily(sub, trend.head(5)["hashtag"].tolist())
            fig = px.line(daily, x="date", y="mentions", color="hashtag", markers=True,
                          title="Daily mentions of the 5 highest-momentum hashtags")
            fig.update_xaxes(tickformat="%b %d", dtick=86400000)
            ui.show(fig, 340)

    ui.section("Hashtag sentiment", "Average VADER score of the tweets using each hashtag. This measures the tone of the wording, "
               "not support for the hashtag's cause (an anti-candidate hashtag can sit in upbeat language).")
    min_t = st.slider("Minimum tweets per hashtag", 5, 200, 20, key="h_minsent")
    hs = an.hashtag_sentiment(fdf, min_t, exclude)
    if len(hs) < 2:
        st.info("Not enough hashtags with that many tweets - lower the minimum.")
    else:
        k = min(10, len(hs) // 2)
        view = pd.concat([hs.head(k).assign(group="Most negative"), hs.tail(k).assign(group="Most positive")])
        fig = px.bar(view, x="avg_sentiment", y="hashtag", color="group", orientation="h",
                     color_discrete_map={"Most negative": "#DC2626", "Most positive": "#16A34A"},
                     hover_data=["tweets"], title="Average VADER sentiment by hashtag")
        fig.update_yaxes(categoryorder="total ascending")
        ui.show(fig, 120 + 26 * len(view))


def tab_emotions(fdf, colors):
    ui.section("Emotion analysis",
               "Plutchik's 8 basic emotions detected with a transparent dictionary (see the word lists below). "
               "A tweet can express several emotions.")
    share = em.emotion_share(fdf)
    reports["emotion_share"] = share.reset_index()
    c1, c2 = st.columns(2)
    with c1:
        fig = px.imshow(share, text_auto=".1f", color_continuous_scale="Purples", aspect="auto",
                        title="% of tweets expressing each emotion", labels=dict(color="%", x="", y=""))
        ui.show(fig, 340)
    with c2:
        long = share.reset_index().melt(id_vars="campaign", var_name="emotion", value_name="pct")
        fig = px.bar(long, x="emotion", y="pct", color="campaign", barmode="group", color_discrete_map=colors,
                     title="Emotion prevalence by campaign", labels={"pct": "% of tweets"})
        ui.show(fig, 340)

    cols = st.columns(max(1, fdf["campaign"].nunique()))
    for col, (campaign, g) in zip(cols, fdf.groupby("campaign", sort=True)):
        dom = g.loc[g["dominant_emotion"] != "None", "dominant_emotion"].value_counts().rename_axis("emotion").reset_index(name="tweets")
        with col:
            if dom.empty:
                st.caption(f"{campaign}: no emotional tweets")
                continue
            fig = px.pie(dom, names="emotion", values="tweets", hole=0.5, color="emotion",
                         color_discrete_map=config.EMOTION_COLORS, title=f"{campaign}: dominant emotion")
            ui.show(fig, 330, legend="right")

    emo = st.selectbox("Emotion to explore", em.EMOTIONS, key="emo_pick")
    c1, c2 = st.columns(2)
    with c1:
        d = (fdf.assign(p=fdf[f"emo_{emo}"] > 0).groupby(["date", "campaign"])["p"].mean().mul(100).rename("pct").reset_index())
        fig = px.line(d, x="date", y="pct", color="campaign", markers=True, color_discrete_map=colors,
                      title=f"Daily % of tweets expressing {emo.lower()}", labels={"pct": "% of tweets"})
        ui.show(fig, 340)
    with c2:
        val = (fdf[fdf["dominant_emotion"] != "None"].groupby("dominant_emotion")["vader_compound"].mean()
               .rename("avg_sentiment").reset_index())
        fig = px.bar(val.sort_values("avg_sentiment"), x="avg_sentiment", y="dominant_emotion", orientation="h",
                     color="avg_sentiment", color_continuous_scale="RdYlGn", range_color=[-0.6, 0.6],
                     title="Sanity check: average VADER sentiment of each dominant emotion")
        fig.update_layout(coloraxis_showscale=False, yaxis_title=None)
        ui.show(fig, 340)

    ui.section(f"Most engaged tweets expressing {emo.lower()}")
    ex = fdf[fdf[f"emo_{emo}"] > 0].nlargest(10, "engagement")
    if len(ex):
        ui.table(ex[["created_at", "campaign", "tweet", "likes", "retweet_count", "vader_label"]])
    else:
        st.caption("No tweets.")
    with st.expander("Show the emotion dictionary used"):
        for name, words in em.EMOTION_LEXICON.items():
            st.markdown(f"**{name}**: " + ", ".join(f"`{w}`" for w in words))


def tab_engagement(fdf, summary, colors):
    ui.section("Engagement", f"Engagement score = likes + {config.RETWEET_WEIGHT} x retweets "
                             "(a retweet reaches a new audience, so it is weighted higher).")
    c = st.columns(max(1, len(summary)))
    for col, (_, r) in zip(c, summary.iterrows()):
        col.metric(f"{r['campaign']}: avg engagement / tweet", f"{r['avg_engagement']:.2f}")
        col.caption(f"Median {r['median_engagement']:.0f}  |  {r['engaged_tweets_pct']:.0f}% of tweets got any engagement")

    ebs = an.engagement_by_sentiment(fdf)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(ebs, x="vader_label", y="avg_engagement", color="campaign", barmode="group", color_discrete_map=colors,
                     category_orders={"vader_label": config.SENTIMENT_ORDER}, hover_data=["tweets", "median_engagement"],
                     title="Average engagement by tweet sentiment", labels={"vader_label": "", "avg_engagement": "Avg engagement"})
        ui.show(fig, 360)
    with c2:
        lz = an.lorenz_curve(fdf)
        fig = px.line(lz, x="tweet_share", y="engagement_share", color="campaign", color_discrete_map=colors,
                      title="Engagement concentration (Lorenz curve)",
                      labels={"tweet_share": "Cumulative % of tweets (least -> most engaged)", "engagement_share": "Cumulative % of engagement"})
        fig.add_shape(type="line", x0=0, y0=0, x1=100, y1=100, line=dict(dash="dot", color="#94A3B8"))
        ui.show(fig, 360)

    c1, c2 = st.columns(2)
    with c1:
        h = an.hourly_activity(fdf)
        fig = px.line(h, x="hour", y="tweets", color="campaign", markers=True, color_discrete_map=colors,
                      title="Tweets by hour of day (as stored in the dataset, UTC)")
        fig.update_xaxes(dtick=3)
        ui.show(fig, 340)
    with c2:
        pts = fdf[(fdf["likes"] > 0) & (fdf["retweet_count"] > 0)]
        pts = pts.sample(min(len(pts), 3000), random_state=1).assign(snippet=lambda d: d["tweet"].str.slice(0, 110))
        if pts.empty:
            st.caption("No tweets with both likes and retweets.")
        else:
            fig = px.scatter(pts, x="likes", y="retweet_count", color="campaign", log_x=True, log_y=True, opacity=0.55,
                             color_discrete_map=colors, hover_data={"snippet": True}, title="Likes vs retweets (log scale, tweets with both)")
            ui.show(fig, 340)

    ui.section("Top tweets")
    c = st.columns(3)
    by = c[0].selectbox("Rank by", ["engagement", "likes", "retweet_count"], key="top_by")
    n = c[1].slider("How many", 5, 50, 10, key="top_n")
    ui.table(an.top_tweets(fdf, by, n).round({"vader_compound": 3}), column_config={"tweet": st.column_config.TextColumn(width="large")})


def tab_geography(fdf):
    ui.section("Geography", "Tweets are mapped by the state field supplied in the dataset.")
    if (fdf["state"] != "").sum() == 0:
        st.info("This dataset has no state information.")
        return
    c = st.columns(3)
    scope = c[0].selectbox("Campaign", ["All selected"] + sorted(fdf["campaign"].unique()), key="geo_scope")
    metric = c[1].selectbox("Colour by", ["tweets", "net_sentiment", "avg_sentiment", "avg_engagement"], key="geo_metric")
    min_t = c[2].slider("Minimum tweets per state", 1, 200, 30, key="geo_min")
    sub = fdf if scope == "All selected" else fdf[fdf["campaign"] == scope]
    ss = an.state_summary(sub, min_t)
    if ss.empty:
        st.info("No state reaches that minimum - lower it.")
        return
    mapped = ss.dropna(subset=["code"]).set_index("code")
    diverging = metric in {"net_sentiment", "avg_sentiment"}
    if not mapped.empty:
        cols, rows = 11, 8
        z = np.full((rows, cols), np.nan)
        base = np.full((rows, cols), np.nan)
        label = np.full((rows, cols), "", dtype=object)
        hover = np.full((rows, cols), "", dtype=object)
        for code, (cx, cy) in config.US_TILE_GRID.items():
            base[cy, cx] = 0
            label[cy, cx] = code
            if code in mapped.index:
                r = mapped.loc[code]
                z[cy, cx] = r[metric]
                hover[cy, cx] = (f"<b>{r['state']}</b><br>Tweets: {int(r['tweets']):,}<br>Net sentiment: {r['net_sentiment']:+.1f} pts"
                                 f"<br>Avg sentiment: {r['avg_sentiment']:+.3f}<br>Avg engagement: {r['avg_engagement']:.2f}")
            else:
                hover[cy, cx] = f"{code}: fewer than {min_t} tweets"
        zmax = float(np.nanmax(np.abs(z))) if diverging else None
        fig = go.Figure()
        fig.add_trace(go.Heatmap(z=base, text=label, texttemplate="%{text}", hovertext=hover, hovertemplate="%{hovertext}<extra></extra>",
                                 colorscale=[[0, "#E2E8F0"], [1, "#E2E8F0"]], showscale=False, xgap=3, ygap=3, hoverongaps=False))
        fig.add_trace(go.Heatmap(z=z, text=label, texttemplate="%{text}", hovertext=hover, hovertemplate="%{hovertext}<extra></extra>",
                                 colorscale="RdYlGn" if diverging else "Blues", zmid=0 if diverging else None,
                                 zmin=-zmax if diverging else None, zmax=zmax if diverging else None,
                                 xgap=3, ygap=3, hoverongaps=False, colorbar=dict(title=metric.replace("_", " "), thickness=14)))
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False, autorange="reversed", scaleanchor="x")
        fig.update_layout(title=f"{metric.replace('_', ' ').title()} by US state (grey = fewer than {min_t} tweets)")
        ui.show(fig, 520, legend="none")
    top = ss.sort_values(metric, ascending=False).head(15)
    fig = px.bar(top, x=metric, y="state", orientation="h", title=f"Top 15 states by {metric.replace('_', ' ')}",
                 color=metric, color_continuous_scale="RdYlGn" if diverging else "Blues")
    fig.update_yaxes(categoryorder="total ascending")
    fig.update_layout(coloraxis_showscale=False)
    ui.show(fig, 480)


def tab_data(fdf, report, df_all, summary, daily):
    ui.section("Data cleaning report", "How the raw files became the analysed dataset.")
    steps = pd.DataFrame(report["steps"], columns=["Step", "Rows remaining", "Rows removed"])
    c1, c2 = st.columns([3, 3])
    with c1:
        ui.table(steps)
        st.caption(f"Files: {', '.join(report['files'])}")
    with c2:
        fig = px.funnel(steps, x="Rows remaining", y="Step", title="Cleaning funnel", color_discrete_sequence=["#6366F1"])
        ui.show(fig, 300)

    ui.section("Downloads", "CSV exports reflect the current sidebar filters and the analysis settings you chose in each tab.")
    export = fdf.assign(hashtags=fdf["hashtags"].map(" ".join)).drop(columns=["date", "hour", "n_hashtags"])
    files = {
        "tweets_scored.csv": export,
        "campaign_summary.csv": summary,
        "daily_sentiment.csv": daily,
        **{f"{k}.csv": v for k, v in reports.items()},
    }
    labels = {
        "tweets_scored.csv": "Tweets + scores",
        "campaign_summary.csv": "Campaign summary",
        "daily_sentiment.csv": "Daily sentiment",
        "top_hashtags.csv": "Top hashtags", "trending_hashtags.csv": "Trending hashtags",
        "emotion_share.csv": "Emotions", "bert_vs_vader.csv": "BERT vs VADER",
    }
    cols = st.columns(3)
    blobs = {}
    for i, (name, frame) in enumerate(files.items()):
        blobs[name] = to_csv(frame)
        with cols[i % 3]:
            ui.download(f"⬇ {labels.get(name, name)} ({len(frame):,} rows)", blobs[name], name, "text/csv", f"dl_{name}")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, blob in blobs.items():
            zf.writestr(name, blob)
    ui.download("⬇ Download everything (ZIP)", buf.getvalue(), "campaign_analytics_reports.zip", "application/zip",
                "dl_zip", primary=True)

    ui.section("Data explorer", "First 500 filtered tweets.")
    ui.table(export[["created_at", "campaign", "tweet", "likes", "retweet_count", "state", "vader_compound",
                     "vader_label", "dominant_emotion"]].head(500).round({"vader_compound": 3}),
             column_config={"tweet": st.column_config.TextColumn(width="large")})

    with st.expander("Methodology & limitations"):
        st.markdown(f"""
- **Dataset**: [US Election 2020 Tweets]({config.KAGGLE_URL}) - tweets containing #Trump / #Biden hashtags, 15 Oct - 8 Nov 2020.
- **Cleaning**: parse timestamps, coerce counts, drop empty rows, optionally remove exact-duplicate texts per campaign
  (keeping the most-engaged copy) and non-English tweets (stop-word/script heuristic - not perfect), strip URLs and @mentions for modelling.
- **VADER**: rule-based sentiment tuned for social media, English only. Neutral band +/-{config.VADER_THRESHOLD}.
- **BERT (optional)**: pretrained Hugging Face model, run on a stratified sample; labels are mapped to Positive / Neutral / Negative.
- **Emotions**: dictionary matching (no sarcasm / context understanding).
- **Engagement**: likes + {config.RETWEET_WEIGHT} x retweets. Follower counts are not used.
- **Trending**: hashtag rate per 1,000 tweets, recent window vs baseline, with smoothing for new hashtags.
- **Limits**: hashtag-collected tweets are not a random sample of voters; sentiment toward a candidate is not the same as support or votes.
""")


# ------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------
files, source_label = dl.find_data_files()
if not files:
    setup_screen()
    st.stop()
is_sample = "sample" in source_label.lower()
signature = dl.file_signature(files)

with st.sidebar:
    st.markdown("## 🗳️ Controls")
    st.caption(f"Data source: **{source_label}**")
    with st.expander("⚙️ Data settings"):
        cap_choice = st.selectbox("Max tweets per campaign", config.ROW_CAP_OPTIONS,
                                  index=config.ROW_CAP_OPTIONS.index(config.DEFAULT_ROW_CAP), disabled=is_sample,
                                  format_func=lambda x: "All" if x == "All" else f"{x:,}",
                                  help="Random sample (seeded) so big files stay fast. 'All' needs several GB of RAM on the full dataset.")
        english_only = st.checkbox("English tweets only (heuristic)", True, help="VADER is English-only.")
        dedupe = st.checkbox("Remove duplicate tweets", True, help="Exact duplicate texts per campaign (retweet/bot copies).")
cap = None if (cap_choice == "All" or is_sample) else int(cap_choice)

try:
    df, report = build_dataset(signature, cap, english_only, dedupe)
except Exception as exc:
    st.error(f"Could not load the data: {exc}")
    st.stop()
data_key = (signature, cap, english_only, dedupe)
reports: dict[str, pd.DataFrame] = {}

with st.sidebar:
    st.markdown("### Filters")
    campaigns = sorted(df["campaign"].unique())
    sel_c = st.multiselect("Campaigns", campaigns, default=campaigns, key="f_campaigns")
    dmin, dmax = df["date"].min().date(), df["date"].max().date()
    rng = st.date_input("Date range", value=(dmin, dmax), min_value=dmin, max_value=dmax, key="f_dates")
    sel_s = st.multiselect("Sentiment (VADER)", config.SENTIMENT_ORDER, default=config.SENTIMENT_ORDER, key="f_sent")
    state_opts = sorted(s for s in df["state"].unique() if s)
    sel_states = st.multiselect("States (empty = all)", state_opts, key="f_states") if state_opts else []
    min_eng = st.number_input("Minimum engagement", 0, int(df["engagement"].max()), 0, key="f_mineng",
                              help=f"likes + {config.RETWEET_WEIGHT} x retweets")
    kw = st.text_input("Tweet contains", key="f_kw", placeholder="e.g. covid")
    tag = st.text_input("Hashtag contains", key="f_tag", placeholder="e.g. maga")
    st.button("↺ Reset filters", on_click=reset_filters)

d0, d1 = (rng if isinstance(rng, (tuple, list)) and len(rng) == 2 else (dmin, dmax))
mask = (df["campaign"].isin(sel_c) & df["vader_label"].isin(sel_s) & (df["engagement"] >= min_eng)
        & df["date"].between(pd.Timestamp(d0), pd.Timestamp(d1)))
if sel_states:
    mask &= df["state"].isin(sel_states)
if kw.strip():
    mask &= df["tweet"].str.contains(kw.strip(), case=False, regex=False)
if tag.strip():
    needle = tag.strip().lstrip("#").lower()
    mask &= df["hashtags"].map(lambda tags: any(needle in t for t in tags))
fdf = df[mask]

with st.sidebar:
    st.success(f"**{len(fdf):,}** of {len(df):,} tweets match")

ui.hero(
    config.APP_TITLE,
    "Compare campaign sentiment, emotions, hashtags and engagement in the 2020 US election conversation.",
    [source_label, f"{len(df):,} tweets analysed", f"{report['date_min']:%d %b} - {report['date_max']:%d %b %Y}",
     " vs ".join(campaigns)],
)
if fdf.empty:
    st.warning("No tweets match the current filters. Widen them or press **Reset filters** in the sidebar.")
    st.stop()

colors = dl.campaign_colors(campaigns)
summary = an.campaign_summary(fdf)
daily = an.daily_sentiment(fdf)

tabs = st.tabs(["📊 Overview", "🧠 Sentiment: VADER vs BERT", "⚖️ Campaign comparison", "#️⃣ Hashtags",
                "🎭 Emotions", "🔥 Engagement", "🗺️ Geography", "📁 Data & reports"])
with tabs[0]:
    tab_overview(fdf, summary, colors)
with tabs[1]:
    tab_sentiment(fdf, colors, data_key)
with tabs[2]:
    tab_comparison(fdf, summary, colors)
with tabs[3]:
    tab_hashtags(fdf, colors)
with tabs[4]:
    tab_emotions(fdf, colors)
with tabs[5]:
    tab_engagement(fdf, summary, colors)
with tabs[6]:
    tab_geography(fdf)
with tabs[7]:
    tab_data(fdf, report, df, summary, daily)
