from __future__ import annotations

import json
import math
import os
import random
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EDA_DIR = PROCESSED_DIR / "eda_strict"
PERSONA_DIR = PROCESSED_DIR / "persona_support"
SIM_DIR = PROCESSED_DIR / "simulation_trial"
EVAL_DIR = PROCESSED_DIR / "evaluation_trial"
REPORT_TABLES_DIR = PROCESSED_DIR / "report_tables"
PROMPTS_DIR = ROOT / "prompts"
REPORT_DIR = ROOT / "report_support"
MPLCONFIGDIR = ROOT / ".mplconfig"

STRICT_DATASET = PROCESSED_DIR / "strict_cleaned_price_reactions.csv"
STRICT_SUMMARY = PROCESSED_DIR / "strict_filter_summary.json"
STRICT_MANIFEST = PROCESSED_DIR / "strict_thread_manifest.csv"

BROAD_DATASET = PROCESSED_DIR / "reddit_streaming_price_ranges_combined.csv"
THREAD_REVIEW = PROCESSED_DIR / "thread_review_autolabeled.csv"

PLATFORM_LABELS = {
    "Netflix": "Netflix",
    "DisneyPlus": "Disney+",
}

PLATFORM_NAMES = {
    "Netflix": ["netflix"],
    "DisneyPlus": ["disney plus", "disney+", "disneyplus", "hulu", "espn+"],
}

PRICE_TERMS = [
    "price",
    "prices",
    "pricing",
    "price hike",
    "price hikes",
    "price increase",
    "price increases",
    "cost",
    "costs",
    "subscription",
    "subscriptions",
    "monthly",
    "fee",
    "plan",
    "plans",
    "bundle",
    "bundles",
    "ad tier",
    "ads",
    "ad-free",
    "ad free",
    "ad-supported",
    "ad supported",
    "hike",
    "hikes",
    "raised",
    "raise",
    "raises",
    "increase",
    "increased",
    "expensive",
    "overpriced",
    "worth",
    "cancel",
    "canceling",
    "cancelling",
]

STRONG_PRICE_TITLE_TERMS = [
    "price hike",
    "price hikes",
    "price increase",
    "price increases",
    "raising prices",
    "price rise",
    "subscription price",
    "subscription prices",
    "plans and prices",
    "hiked its subscription prices",
    "constant price hikes",
    "ad-supported tier",
    "ad supported tier",
    "bundle pricing",
]

OFFTOPIC_TITLE_TERMS = [
    "tech support",
    "missing content",
    "watchlist",
    "black friday",
    "mega thread",
    "what should i watch",
    "recommend",
    "season",
    "episode",
    "cast",
    "deserved way more attention",
    "show you think",
    "watch right now",
]

LOW_PRECISION_TITLE_TERMS = [
    "gift card",
    "jimmy kimmel",
    "directv",
    "fox one",
    "nfl cost",
    "comcast",
    "peacock",
]

REACTION_TERMS = [
    "too expensive",
    "not worth",
    "worth it",
    "cancel",
    "canceled",
    "cancelled",
    "keeping",
    "downgrade",
    "ads",
    "ad-free",
    "bundle",
    "cheaper",
    "same price",
    "paying",
    "staying",
    "leaving",
    "switching",
    "pirate",
    "drop",
]

REACTION_LABEL_CUES = {
    "cancel_or_leave": [
        "cancel",
        "cancelled",
        "canceled",
        "unsubscribe",
        "unsubscribed",
        "drop it",
        "dropping",
        "done with",
        "not worth",
        "pirate",
    ],
    "ad_tier_or_bundle_concern": [
        "ads",
        "ad-free",
        "ad free",
        "ad tier",
        "bundle",
        "premium",
        "downgrade",
    ],
    "value_comparison": [
        "same price",
        "cheaper",
        "better value",
        "worth it",
        "worth",
        "cable",
        "compare",
        "price for",
        "monthly",
    ],
    "frustrated_but_still_engaged": [
        "annoying",
        "ridiculous",
        "greedy",
        "too expensive",
        "price hike",
        "raise prices",
        "fed up",
        "still have",
        "still keep",
    ],
}

STOPWORDS = {
    "a",
    "about",
    "after",
    "again",
    "all",
    "also",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "but",
    "by",
    "can",
    "cant",
    "could",
    "did",
    "didnt",
    "do",
    "does",
    "doesnt",
    "doing",
    "dont",
    "down",
    "even",
    "for",
    "from",
    "get",
    "got",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "him",
    "his",
    "how",
    "i",
    "id",
    "if",
    "im",
    "in",
    "into",
    "is",
    "isnt",
    "it",
    "its",
    "ive",
    "just",
    "like",
    "me",
    "more",
    "most",
    "my",
    "no",
    "not",
    "now",
    "of",
    "on",
    "one",
    "only",
    "or",
    "our",
    "out",
    "people",
    "really",
    "so",
    "some",
    "still",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "to",
    "too",
    "up",
    "us",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "which",
    "who",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
    "netflix",
    "disney",
    "disneyplus",
    "disney+",
    "hulu",
    "espn",
    "espn+",
}

PERSONA_LEXICONS = {
    "price_sensitive_canceller": [
        "cancel",
        "cancelled",
        "canceled",
        "too expensive",
        "not worth",
        "done",
        "dropping",
        "unsubscribe",
        "unsubscribed",
        "pirate",
    ],
    "frustrated_loyal_subscriber": [
        "years",
        "been with",
        "still have",
        "still keep",
        "loyal",
        "used to love",
        "frustrated",
        "fed up",
        "annoying",
        "sucks",
    ],
    "ad_tier_bundle_skeptic": [
        "ads",
        "ad-free",
        "ad free",
        "ad tier",
        "bundle",
        "tier",
        "downgrade",
        "premium",
    ],
    "value_for_money_comparer": [
        "worth",
        "value",
        "cheaper",
        "same price",
        "compare",
        "better",
        "cable",
        "budget",
        "save money",
        "cost efficient",
    ],
}


def ensure_output_dirs() -> None:
    for path in [
        PROCESSED_DIR,
        EDA_DIR,
        PERSONA_DIR,
        SIM_DIR,
        EVAL_DIR,
        REPORT_TABLES_DIR,
        PROMPTS_DIR,
        REPORT_DIR,
        MPLCONFIGDIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def configure_matplotlib() -> None:
    ensure_output_dirs()
    os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
    os.environ.setdefault("MPLBACKEND", "Agg")


def load_csv(path: Path | str) -> pd.DataFrame:
    return pd.read_csv(path)


def parse_timestamp(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def deduplicate_comments(df: pd.DataFrame) -> pd.DataFrame:
    dedup_cols = ["platform_name", "subreddit", "thread_id", "comment_id"]
    existing = [col for col in dedup_cols if col in df.columns]
    return df.drop_duplicates(subset=existing).copy()


def normalize_text(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"http\\S+", " ", text)
    text = re.sub(r"[^a-z0-9+\\s]", " ", text)
    text = re.sub(r"\\s+", " ", text)
    return text.strip()


def word_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z][a-z0-9+']+", normalize_text(text))


def word_count(text: str) -> int:
    return len(word_tokens(text))


def contains_any(text: str, terms: Iterable[str]) -> bool:
    text_l = str(text or "").lower()
    return any(term in text_l for term in terms)


def count_terms(text: str, terms: Iterable[str]) -> int:
    text_l = str(text or "").lower()
    return sum(1 for term in terms if term in text_l)


def top_words(series: pd.Series, limit: int = 50) -> pd.DataFrame:
    counter: Counter[str] = Counter()
    for text in series.fillna(""):
        for token in word_tokens(text):
            if token in STOPWORDS or token.isdigit() or len(token) < 3:
                continue
            counter[token] += 1
    rows = [{"word": word, "count": count} for word, count in counter.most_common(limit)]
    return pd.DataFrame(rows)


def sample_stratified(
    df: pd.DataFrame,
    group_cols: list[str],
    target_n: int,
    random_seed: int = 42,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    random.seed(random_seed)
    groups = list(df.groupby(group_cols, dropna=False))
    base_n = max(1, math.floor(target_n / max(1, len(groups))))
    parts: list[pd.DataFrame] = []

    for _, group in groups:
        n = min(len(group), base_n)
        parts.append(group.sample(n=n, random_state=random_seed))

    sampled = pd.concat(parts, ignore_index=True).drop_duplicates()
    remaining = target_n - len(sampled)
    if remaining > 0:
        remainder = df.drop(sampled.index, errors="ignore")
        if not remainder.empty:
            sampled = pd.concat(
                [
                    sampled,
                    remainder.sample(n=min(remaining, len(remainder)), random_state=random_seed),
                ],
                ignore_index=True,
            ).drop_duplicates()

    return sampled.head(target_n).copy()


def persona_scores(text: str) -> dict[str, int]:
    text_l = normalize_text(text)
    return {
        persona: sum(1 for cue in cues if cue in text_l)
        for persona, cues in PERSONA_LEXICONS.items()
    }


def dominant_persona_label(text: str) -> str:
    scores = persona_scores(text)
    best = max(scores.items(), key=lambda item: item[1])
    if best[1] <= 0:
        return "unclear"
    return best[0]


def sentiment_bucket(text: str) -> str:
    text_l = normalize_text(text)
    negative = count_terms(
        text_l,
        [
            "hate",
            "awful",
            "bad",
            "ridiculous",
            "greedy",
            "annoying",
            "sucks",
            "worse",
            "expensive",
            "cancel",
            "not worth",
        ],
    )
    positive = count_terms(
        text_l,
        [
            "fine",
            "fair",
            "worth",
            "okay",
            "good",
            "deal",
            "reasonable",
            "cheap",
            "better",
        ],
    )
    if negative > positive:
        return "negative"
    if positive > negative:
        return "positive"
    return "mixed_or_neutral"


def stance_bucket(text: str) -> str:
    text_l = normalize_text(text)
    if contains_any(
        text_l,
        [
            "cancel",
            "not worth",
            "too expensive",
            "greedy",
            "price hike",
            "raise prices",
            "dropping",
            "done with",
        ],
    ):
        return "opposed"
    if contains_any(
        text_l,
        [
            "worth it",
            "still paying",
            "fine with",
            "i will keep",
            "reasonable",
            "fair",
        ],
    ):
        return "accepting"
    return "neutral_or_information"


def reaction_label(text: str) -> str:
    text_l = normalize_text(text)
    for label in [
        "cancel_or_leave",
        "ad_tier_or_bundle_concern",
        "value_comparison",
        "frustrated_but_still_engaged",
    ]:
        if contains_any(text_l, REACTION_LABEL_CUES[label]):
            return label
    return "neutral_or_information"


def generation_quality_note(text: str) -> str:
    text = str(text or "").strip()
    lower = text.lower()
    notes = []
    if any(phrase in lower for phrase in ["i'm sorry", "please feel free", "let me know", "hey there", "hello"]):
        notes.append("too polite / assistant-like")
    if "#" in text:
        notes.append("contains hashtag-style flair")
    if "?" in text:
        notes.append("question-led rather than reaction-led")
    if len(text.split()) > 40:
        notes.append("longer than target short-comment style")
    if not notes:
        return "reasonably Reddit-like short reaction"
    return "; ".join(notes)


def save_json(path: Path | str, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def platform_display_name(platform_name: str) -> str:
    return PLATFORM_LABELS.get(platform_name, platform_name)
