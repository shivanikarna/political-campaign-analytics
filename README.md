# 🗳️ Political Campaign Analytics Platform

An interactive Streamlit dashboard that compares campaigns in the **2020 US election Twitter conversation**:
sentiment (VADER, optionally vs BERT), emotions, trending hashtags, engagement, geography and downloadable CSV reports.
Every chart and number is computed from the dataset - nothing is hard-coded.

**Stack:** Python 3.12 · Streamlit · Pandas · Plotly · VADER · (optional) Hugging Face Transformers

## Quick start (Windows PowerShell)

```powershell
cd political-campaign-analytics
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

The app opens at <http://localhost:8501>. It starts immediately on the **bundled sample** (8,000 real tweets).

> If PowerShell blocks `Activate.ps1` ("running scripts is disabled"), run this once in that window:
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and activate again.

## Use the full Kaggle dataset

Dataset: [US Election 2020 Tweets](https://www.kaggle.com/datasets/manchunhui/us-election-2020-tweets)
(`hashtag_donaldtrump.csv` + `hashtag_joebiden.csv`, ~1.7M tweets collected 15 Oct - 8 Nov 2020).

**Option A - manual (no extra install):** sign in to Kaggle, click *Download*, unzip, and copy the two CSV files into the
project's `data/` folder. Restart the app. The sidebar will say *Kaggle dataset (data/)*.

**Option B - script:**
```powershell
pip install kaggle
# Kaggle > Settings > "Create New Token" -> save kaggle.json to %USERPROFILE%\.kaggle\
python scripts/download_data.py
```

Notes
- Files in `data/` take priority over `data/sample/`. The app also accepts any CSV with `created_at`, `tweet`, `likes`,
  `retweet_count` and a `candidate` column (pre-merged copies of the dataset).
- By default each campaign is randomly sampled to **25,000 tweets** (seeded, reproducible) so it stays fast. Change this under
  *Data settings* in the sidebar; "All" needs roughly 3-6 GB of RAM on the full 1.7M rows.
- A tweet containing both #Trump and #Biden appears in both campaigns (that's how the dataset was collected).

## Optional: BERT comparison

The *Sentiment: VADER vs BERT* tab works only if the extra packages are installed; without them it shows a notice and
everything else works normally.

```powershell
pip install -r requirements-bert.txt
streamlit run app.py
```

Choose a model, choose how many tweets to score (stratified by campaign) and press **Run BERT**. The model downloads on the
first run (260-670 MB) and needs internet access once. You then get agreement %, Cohen's kappa, a VADER-vs-BERT confusion
matrix, net sentiment per campaign under both models, and a table of the tweets where they disagree most confidently.
Models: Twitter-RoBERTa (default, tweet-trained), DistilBERT SST-2, multilingual BERT (stars mapped to 3 classes).

## What's in the dashboard

| Tab | What it shows | How it's computed |
|---|---|---|
| Overview | Interactive KPI cards (combined / per campaign with deltas), auto-written insights, volume & sentiment over time | `src/analytics.py` |
| Sentiment: VADER vs BERT | Sentiment mix, score distribution, daily net sentiment, model comparison | VADER compound, ±0.05 neutral band |
| Campaign comparison | Metric picker, radar profile, bootstrap 95% CI for the sentiment gap, emotions, full metric table | 500-resample bootstrap on mean VADER |
| Hashtags | Top hashtags, **trending detection**, hashtag sentiment | lift = recent rate per 1k tweets ÷ baseline rate (smoothed) |
| Emotions | 8 Plutchik emotions per campaign, daily trend, dominant emotion, example tweets | transparent word lists in `src/emotions.py` |
| Engagement | Engagement by sentiment, Lorenz concentration curve, hourly activity, likes vs retweets, top tweets | score = likes + 2 × retweets |
| Geography | Offline US tile-grid map + ranking by tweets / sentiment / engagement | state field in the data |
| Data & reports | Cleaning funnel, data explorer, **7 CSV downloads + ZIP** | respects all filters |

**Sidebar filters:** campaign, date range, sentiment, state, minimum engagement, tweet text, hashtag - plus *Reset filters*.

## Data cleaning

Parse timestamps and counts → drop empty/invalid rows → remove exact duplicate texts per campaign (keeps the most-engaged copy)
→ seeded sampling → strip URLs/@mentions/HTML entities for modelling → optional English-only filter (stop-word and script
heuristic) → extract unique hashtags per tweet. The *Data & reports* tab shows how many rows each step removed.

## Project layout

```
app.py                  Streamlit UI (one function per tab)
src/config.py           paths, colours, model list, lookup tables
src/data_loader.py      file discovery, reading, cleaning
src/sentiment.py        VADER + optional BERT
src/emotions.py         emotion lexicon + scoring
src/analytics.py        metrics, hashtag trends, statistics
src/ui.py               CSS and chart styling
scripts/download_data.py  Kaggle API downloader
tests/test_platform.py  29 tests (logic + full UI via Streamlit AppTest)
data/sample/            8,000-tweet real sample
```

## Tests

```powershell
python -m unittest discover -s tests -v
```

## Limitations (please read)

- Tweets were collected by hashtag; they are **not a representative sample of voters**, and sentiment toward a candidate is not
  the same as support or votes.
- VADER is English-only and rule-based (sarcasm is missed). The emotion lexicon is a dictionary, not a trained model.
- Hashtag sentiment measures the tone of the wording, not agreement with the hashtag's cause.
- The English filter is a heuristic and can misclassify very short tweets.
- Engagement ignores follower counts, and the 2× retweet weight is a documented design choice, not a standard.
- The bundled sample is a slice of the Kaggle data taken from a third-party mirror; check the dataset's licence on Kaggle before
  redistributing it.

## Verification status

Tested on Linux with Python 3.12 (pandas 3.0 and 2.2): all 29 tests pass, the server starts, and every tab was inspected in
headless Chromium. The full 332k-tweet real subset processes in ~50 s using <800 MB RAM. **Not yet run by the author:** a real BERT
model download/inference, the Kaggle API download script's success path, and Windows itself.
