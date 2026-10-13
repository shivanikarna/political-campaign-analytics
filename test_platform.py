"""Run with:  python -m unittest discover -s tests -v

Covers the cleaning pipeline, VADER, emotions, hashtag analytics, the optional BERT
plumbing and the full Streamlit UI (via Streamlit's built-in AppTest harness).
Synthetic frames are used ONLY here, to check maths against known answers.
"""
from __future__ import annotations

import re
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import analytics as an  # noqa: E402
from src import config, emotions as em, sentiment as sm  # noqa: E402
from src import data_loader as dl  # noqa: E402

SAMPLE = sorted(config.SAMPLE_DIR.glob("*.csv"))


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, cls.report = dl.load_clean(SAMPLE, max_rows_per_campaign=None)

    def test_columns_and_types(self):
        for col in ["created_at", "tweet", "text", "campaign", "likes", "retweet_count", "engagement", "hashtags", "date", "hour"]:
            self.assertIn(col, self.df.columns)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(self.df["created_at"]))
        self.assertFalse(self.df["created_at"].isna().any())
        self.assertTrue((self.df["likes"] >= 0).all())

    def test_engagement_formula(self):
        expected = self.df["likes"] + config.RETWEET_WEIGHT * self.df["retweet_count"]
        pd.testing.assert_series_equal(self.df["engagement"], expected, check_names=False)

    def test_no_duplicates_and_campaigns(self):
        self.assertFalse(self.df.duplicated(["campaign", "tweet"]).any())
        self.assertEqual(set(self.df["campaign"]), {"Trump", "Biden"})

    def test_text_cleaning(self):
        self.assertFalse(self.df["text"].str.contains(r"https?://|@\w+|#").any())
        self.assertFalse((self.df["text"] == "").any())

    def test_funnel_adds_up(self):
        steps = self.report["steps"]
        for (_, remaining_prev, _), (_, remaining, removed) in zip(steps, steps[1:]):
            self.assertEqual(remaining_prev - removed, remaining)
        self.assertEqual(steps[-1][1], len(self.df))

    def test_sampling_cap(self):
        small, _ = dl.load_clean(SAMPLE, max_rows_per_campaign=500, english_only=False)
        self.assertTrue((small.groupby("campaign").size() <= 500).all())
        again, _ = dl.load_clean(SAMPLE, max_rows_per_campaign=500, english_only=False)
        pd.testing.assert_frame_equal(small, again)  # seeded -> reproducible

    def test_english_filter(self):
        self.assertTrue(dl.looks_english("Biden is going to win this election and I am happy"))
        self.assertFalse(dl.looks_english("Contra todo pronóstico, el presidente salió de hospital con una sonrisa"))
        self.assertFalse(dl.looks_english("65 Dinge, die Sie nicht über Joe Biden wissen und das ist gut"))

    def test_raw_kaggle_layout(self):
        """Official Kaggle files are wide, CRLF-terminated, named hashtag_<candidate>.csv."""
        header = ("created_at,tweet_id,tweet,likes,retweet_count,source,user_id,user_name,user_screen_name,"
                  "user_description,user_join_date,user_followers_count,user_location,lat,long,city,country,"
                  "continent,state,state_code,collected_at")
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for fname, camp in [("hashtag_donaldtrump.csv", "Trump"), ("hashtag_joebiden.csv", "Biden")]:
                rows = pd.read_csv(SAMPLE[0], lineterminator="\n")
                rows = rows[rows["candidate"].str.contains(camp.lower())].head(50)
                p = Path(tmp) / fname
                with open(p, "w", encoding="utf-8", newline="") as fh:
                    fh.write(header + "\r\n")
                    for _, r in rows.iterrows():
                        tweet = '"' + r["tweet"].replace('"', '""') + '"'
                        fh.write(f'{r["created_at"]},1.3e18,{tweet},{r["likes"]},{r["retweet_count"]},web,1,u,u,,2010,5,,,,,'
                                 f'{r["country"]},{r["continent"]},{r["state"]},,2020-11-09 00:00:00\r\n')
                paths.append(p)
            df, _ = dl.load_clean(paths, max_rows_per_campaign=None, english_only=False)
            self.assertEqual(set(df["campaign"]), {"Trump", "Biden"})
            self.assertEqual(len(df), len(df.drop_duplicates(["campaign", "tweet"])))
            self.assertGreater(len(df), 60)


class TestSentimentAndEmotions(unittest.TestCase):
    def test_vader_direction(self):
        df = pd.DataFrame({"text": ["I love this wonderful great candidate!", "I hate this terrible corrupt liar.", "The debate is on Thursday."]})
        out = sm.add_vader(df)
        self.assertEqual(out["vader_label"].tolist(), ["Positive", "Negative", "Neutral"])
        self.assertTrue(out["vader_compound"].between(-1, 1).all())

    def test_emotions(self):
        df = pd.DataFrame({"text": ["I am furious, this is an outrage", "so scared and afraid of the chaos",
                                    "what a happy wonderful victory", "disgusting liar and fraud",
                                    "wow what a shocking surprise", "nothing to see"]})
        out = em.add_emotions(df)
        self.assertEqual(out["dominant_emotion"].tolist(), ["Anger", "Fear", "Joy", "Disgust", "Surprise", "None"])

    def test_emotion_share_bounds(self):
        df = sm.add_vader(dl.load_clean(SAMPLE, max_rows_per_campaign=None)[0])
        share = em.emotion_share(em.add_emotions(df))
        self.assertTrue(((share >= 0) & (share <= 100)).all().all())

    def test_bert_label_mapping(self):
        cases = {"POSITIVE": "Positive", "negative": "Negative", "neutral": "Neutral", "LABEL_0": "Negative",
                 "LABEL_2": "Positive", "1 star": "Negative", "3 stars": "Neutral", "5 stars": "Positive"}
        for raw, want in cases.items():
            self.assertEqual(sm.normalize_label(raw), want)
        with self.assertRaises(ValueError):
            sm.normalize_label("banana")

    def test_bert_predict_batches(self):
        fake = lambda batch, batch_size: [{"label": "POSITIVE", "score": 0.9} for _ in batch]  # noqa: E731
        seen = []
        out = sm.bert_predict(fake, ["a", "b", "c", ""], batch_size=2, progress=seen.append)
        self.assertEqual(len(out), 4)
        self.assertEqual(seen[-1], 1.0)

    def test_kappa(self):
        a = pd.Series(["Positive", "Negative", "Neutral", "Positive"])
        self.assertAlmostEqual(an.cohen_kappa(a, a), 1.0)
        self.assertLess(an.cohen_kappa(a, pd.Series(["Negative", "Positive", "Positive", "Neutral"])), 0.1)


class TestAnalytics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df, _ = dl.load_clean(SAMPLE, max_rows_per_campaign=None)
        cls.df = em.add_emotions(sm.add_vader(df))

    def test_summary_matches_raw_counts(self):
        s = an.campaign_summary(self.df).set_index("campaign")
        for campaign, n in self.df["campaign"].value_counts().items():
            self.assertEqual(s.loc[campaign, "tweets"], n)
            self.assertAlmostEqual(s.loc[campaign, "positive_pct"] + s.loc[campaign, "neutral_pct"] + s.loc[campaign, "negative_pct"], 100, places=2)
        self.assertEqual(s["total_engagement"].sum(), self.df["engagement"].sum())

    def test_trending_logic_known_answer(self):
        days = pd.date_range("2020-10-01", periods=6)
        rows = []
        for d in days[:3]:  # baseline: #old only
            rows += [dict(date=d, campaign="X", hashtags=["#old"], vader_compound=0.0, engagement=0)] * 100
        for d in days[3:]:  # recent: #old steady + #new appears
            rows += [dict(date=d, campaign="X", hashtags=["#old"], vader_compound=0.0, engagement=0)] * 70
            rows += [dict(date=d, campaign="X", hashtags=["#new"], vader_compound=0.5, engagement=0)] * 30
        trend, (w0, w1) = an.trending_hashtags(pd.DataFrame(rows), recent_days=3, min_recent=10)
        self.assertEqual(trend.iloc[0]["hashtag"], "#new")
        self.assertEqual(trend.iloc[0]["status"], "New")
        old = trend[trend["hashtag"] == "#old"].iloc[0]
        self.assertLess(old["lift"], 1.0)
        self.assertEqual((w0, w1), (days[3], days[5]))

    def test_trending_real_data_runs(self):
        trend, _ = an.trending_hashtags(self.df, 3, 5, config.CORE_HASHTAGS)
        self.assertFalse(trend.empty)
        self.assertTrue((trend["lift"] > 0).all())

    def test_top_hashtags_excludes_core(self):
        top = an.top_hashtags(self.df, 10, config.CORE_HASHTAGS)
        self.assertFalse(top["hashtag"].isin(config.CORE_HASHTAGS).any())

    def test_lorenz_monotonic(self):
        lz = an.lorenz_curve(self.df)
        for _, g in lz.groupby("campaign"):
            self.assertTrue(g["engagement_share"].is_monotonic_increasing)
            self.assertAlmostEqual(g["engagement_share"].iloc[-1], 100, places=3)

    def test_bootstrap_detects_known_gap(self):
        rng = np.random.default_rng(0)
        diff, lo, hi = an.bootstrap_mean_diff(rng.normal(0.3, 0.4, 2000), rng.normal(0.0, 0.4, 2000), n_boot=300)
        self.assertGreater(lo, 0)
        self.assertAlmostEqual(diff, 0.3, delta=0.05)

    def test_tile_grid_covers_every_state_once(self):
        self.assertEqual(set(config.US_TILE_GRID), set(config.US_STATE_CODES.values()))
        self.assertEqual(len(set(config.US_TILE_GRID.values())), len(config.US_TILE_GRID))  # no overlapping tiles

    def test_state_summary_codes(self):
        ss = an.state_summary(self.df, min_tweets=1)
        self.assertTrue(ss["code"].notna().any())


def matched(at) -> int:
    """Number of tweets reported by the sidebar's '**N** of M tweets match' box."""
    return int(re.search(r"\*\*([\d,]+)\*\* of", at.sidebar.success[0].value).group(1).replace(",", ""))


class TestStreamlitApp(unittest.TestCase):
    """Drives the real app headlessly, tab by tab."""

    @classmethod
    def setUpClass(cls):
        from streamlit.testing.v1 import AppTest
        cls.AppTest = AppTest

    def fresh(self):
        return self.AppTest.from_file(str(ROOT / "app.py"), default_timeout=240).run()

    def test_launches_clean(self):
        at = self.fresh()
        self.assertEqual([e.value for e in at.exception], [])
        self.assertEqual(len(at.tabs), 8)
        self.assertGreaterEqual(len(at.get("plotly_chart")), 20)

    def test_campaign_filter_changes_results(self):
        at = self.fresh()
        both = matched(at)
        at.sidebar.multiselect(key="f_campaigns").set_value(["Biden"]).run()
        self.assertEqual([e.value for e in at.exception], [])
        biden = matched(at)
        self.assertLess(biden, both)
        self.assertEqual(at.metric[0].value, f"{biden:,}")  # KPI card reflects the filter

    def test_keyword_and_hashtag_filters(self):
        at = self.fresh()
        at.sidebar.text_input(key="f_kw").set_value("covid").run()
        n = matched(at)
        self.assertEqual([e.value for e in at.exception], [])
        self.assertGreater(n, 0)
        self.assertLess(n, 7000)
        at.sidebar.text_input(key="f_kw").set_value("zzzzqqqq-no-such-text").run()
        self.assertEqual([e.value for e in at.exception], [])
        self.assertTrue(any("No tweets match" in w.value for w in at.warning))

    def test_sentiment_filter_and_min_engagement(self):
        at = self.fresh()
        at.sidebar.multiselect(key="f_sent").set_value(["Negative"]).run()
        at.sidebar.number_input(key="f_mineng").set_value(5).run()
        self.assertEqual([e.value for e in at.exception], [])

    def test_download_buttons_match_filtered_rows(self):
        at = self.fresh()
        at.sidebar.multiselect(key="f_campaigns").set_value(["Trump"]).run()
        n = matched(at)
        labels = [b.proto.label for b in at.get("download_button")]
        tweets_label = next(l for l in labels if "Tweets + scores" in l)
        self.assertIn(f"{n:,} rows", tweets_label)
        self.assertTrue(any("ZIP" in l for l in labels))

    def test_interactive_controls_in_every_tab(self):
        at = self.fresh()
        at.radio(key="kpi_view").set_value("Per campaign").run()
        at.selectbox(key="cmp_metric").set_value("Total engagement").run()
        at.selectbox(key="emo_pick").set_value("Fear").run()
        at.selectbox(key="top_by").set_value("likes").run()
        at.selectbox(key="geo_metric").set_value("net_sentiment").run()
        at.checkbox(key="h_core").set_value(False).run()
        at.slider(key="t_days").set_value(5).run()
        self.assertEqual([e.value for e in at.exception], [])

    def test_bert_section_degrades_gracefully(self):
        if sm.bert_available():
            self.skipTest("BERT installed in this environment")
        at = self.fresh()
        self.assertTrue(any("optional" in i.value for i in at.info))


if __name__ == "__main__":
    unittest.main(verbosity=2)
