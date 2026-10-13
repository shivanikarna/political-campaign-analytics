"""Central configuration: paths, colours, model choices and lookup tables."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SAMPLE_DIR = DATA_DIR / "sample"

APP_TITLE = "Political Campaign Analytics"
KAGGLE_DATASET = "manchunhui/us-election-2020-tweets"
KAGGLE_URL = "https://www.kaggle.com/datasets/manchunhui/us-election-2020-tweets"

# --- Campaign naming / colours -------------------------------------------------
# Raw Kaggle files are named hashtag_donaldtrump.csv / hashtag_joebiden.csv.
# Pre-merged mirrors use a "candidate" column with values trump_df / biden_df.
CAMPAIGN_KEYWORDS = {
    "trump": "Trump",
    "biden": "Biden",
}
CAMPAIGN_COLORS = {"Trump": "#DC2626", "Biden": "#2563EB"}
FALLBACK_COLORS = ["#7C3AED", "#059669", "#D97706", "#0891B2", "#DB2777"]

EMOTION_COLORS = {
    "Anger": "#DC2626", "Fear": "#7C3AED", "Disgust": "#65A30D", "Sadness": "#2563EB",
    "Joy": "#F59E0B", "Trust": "#0D9488", "Surprise": "#EC4899", "Anticipation": "#F97316",
}

SENTIMENT_ORDER = ["Positive", "Neutral", "Negative"]
SENTIMENT_COLORS = {"Positive": "#16A34A", "Neutral": "#94A3B8", "Negative": "#DC2626"}

# VADER's authors recommend +/-0.05 as the neutral band.
VADER_THRESHOLD = 0.05

# Engagement score = likes + RETWEET_WEIGHT * retweets  (a retweet spreads the
# message to a new audience, so it is weighted higher than a like).
RETWEET_WEIGHT = 2

# --- Optional BERT-family models ------------------------------------------------
BERT_MODELS = {
    "cardiffnlp/twitter-roberta-base-sentiment-latest": (
        "Twitter-RoBERTa (3-class, trained on tweets) - recommended, ~500 MB"
    ),
    "distilbert-base-uncased-finetuned-sst-2-english": (
        "DistilBERT SST-2 (binary, lightweight) - ~260 MB"
    ),
    "nlptown/bert-base-multilingual-uncased-sentiment": (
        "BERT multilingual (1-5 stars mapped to 3 classes) - ~670 MB"
    ),
}
DEFAULT_BERT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Hashtags that simply name the campaign; hidden by default in hashtag views.
CORE_HASHTAGS = {
    "#trump", "#biden", "#joebiden", "#donaldtrump", "#realdonaldtrump",
    "#election2020", "#elections2020", "#2020election",
}

# Cap options for the sidebar (rows per campaign analysed).
ROW_CAP_OPTIONS = [5_000, 10_000, 25_000, 50_000, 100_000, "All"]
DEFAULT_ROW_CAP = 25_000

US_STATE_CODES = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI",
    "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI",
    "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX",
    "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}

# Tile-grid map: one square per state, (column, row). Works offline - no GeoJSON/CDN needed.
US_TILE_GRID = {
    "AK": (0, 0), "ME": (10, 0),
    "VT": (9, 1), "NH": (10, 1),
    "WA": (0, 2), "ID": (1, 2), "MT": (2, 2), "ND": (3, 2), "MN": (4, 2), "IL": (5, 2), "WI": (6, 2),
    "MI": (7, 2), "NY": (8, 2), "RI": (9, 2), "MA": (10, 2),
    "OR": (0, 3), "NV": (1, 3), "WY": (2, 3), "SD": (3, 3), "IA": (4, 3), "IN": (5, 3), "OH": (6, 3),
    "PA": (7, 3), "NJ": (8, 3), "CT": (9, 3),
    "CA": (0, 4), "UT": (1, 4), "CO": (2, 4), "NE": (3, 4), "MO": (4, 4), "KY": (5, 4), "WV": (6, 4),
    "VA": (7, 4), "MD": (8, 4), "DE": (9, 4),
    "AZ": (1, 5), "NM": (2, 5), "KS": (3, 5), "AR": (4, 5), "TN": (5, 5), "NC": (6, 5), "SC": (7, 5),
    "DC": (8, 5),
    "OK": (3, 6), "LA": (4, 6), "MS": (5, 6), "AL": (6, 6), "GA": (7, 6),
    "HI": (0, 7), "TX": (3, 7), "FL": (8, 7),
}
