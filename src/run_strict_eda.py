from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from capstone_pipeline_utils import (
    EDA_DIR,
    PERSONA_DIR,
    STRICT_DATASET,
    STRICT_SUMMARY,
    configure_matplotlib,
    deduplicate_comments,
    ensure_output_dirs,
    load_csv,
    parse_timestamp,
    platform_display_name,
    sample_stratified,
    save_json,
    top_words,
    word_count,
)


def save_plot_platform_counts(counts: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts["platform_name"], counts["count"], color=["#111827", "#0f766e"])
    ax.set_title("Strict Dataset Rows by Platform")
    ax.set_xlabel("Platform")
    ax.set_ylabel("Rows")
    fig.tight_layout()
    fig.savefig(EDA_DIR / "platform_counts.png", dpi=180)
    plt.close(fig)


def save_plot_monthly_counts(monthly_df: pd.DataFrame, platform_name: str) -> None:
    import matplotlib.pyplot as plt

    subset = monthly_df[monthly_df["platform_name"].eq(platform_name)].copy()
    subset["month"] = pd.to_datetime(subset["month"])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(subset["month"], subset["count"], marker="o", color="#0f766e" if platform_name == "DisneyPlus" else "#111827")
    ax.set_title(f"Monthly Comment Counts: {platform_display_name(platform_name)}")
    ax.set_xlabel("Month")
    ax.set_ylabel("Rows")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(EDA_DIR / f"monthly_counts_{platform_name}.png", dpi=180)
    plt.close(fig)


def save_plot_length_histogram(df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["word_count"], bins=30, color="#2563eb", edgecolor="white")
    ax.set_title("Comment Length Distribution")
    ax.set_xlabel("Word count")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(EDA_DIR / "comment_length_histogram.png", dpi=180)
    plt.close(fig)


def build_eda_outputs(dataset_path: Path = STRICT_DATASET) -> dict:
    configure_matplotlib()
    ensure_output_dirs()

    df = deduplicate_comments(load_csv(dataset_path))
    df["timestamp"] = parse_timestamp(df["timestamp"])
    df["text"] = df["text"].fillna("").astype(str)
    df["word_count"] = df["text"].map(word_count)
    df["month"] = df["timestamp"].dt.to_period("M").astype(str)

    total_rows = int(len(df))
    unique_threads = int(df["thread_id"].nunique())
    unique_comments = int(df["comment_id"].nunique())

    platform_counts = df.groupby("platform_name").size().reset_index(name="count")
    subreddit_counts = df.groupby("subreddit").size().reset_index(name="count").sort_values("count", ascending=False)
    source_type_counts = df.groupby("source_type").size().reset_index(name="count").sort_values("count", ascending=False)
    monthly_counts = (
        df.groupby(["platform_name", "month"])
        .size()
        .reset_index(name="count")
        .sort_values(["platform_name", "month"])
    )

    top_words_all = top_words(df["text"], limit=75)
    top_words_by_platform: dict[str, pd.DataFrame] = {}
    for platform_name, group in df.groupby("platform_name"):
        top_words_by_platform[platform_name] = top_words(group["text"], limit=60)

    platform_counts.to_csv(EDA_DIR / "platform_counts.csv", index=False)
    subreddit_counts.to_csv(EDA_DIR / "subreddit_counts.csv", index=False)
    source_type_counts.to_csv(EDA_DIR / "source_type_counts.csv", index=False)
    monthly_counts.to_csv(EDA_DIR / "monthly_counts.csv", index=False)
    top_words_all.to_csv(EDA_DIR / "top_words.csv", index=False)

    for platform_name, words_df in top_words_by_platform.items():
        words_df.to_csv(EDA_DIR / f"top_words_{platform_name}.csv", index=False)

    df["length_bucket"] = pd.cut(
        df["word_count"],
        bins=[0, 10, 25, 50, 100, 1000],
        labels=["1-10", "11-25", "26-50", "51-100", "100+"],
        include_lowest=True,
    )
    sample_for_personas = sample_stratified(
        df[df["text"].str.strip().ne("")].copy(),
        group_cols=["platform_name", "source_type", "length_bucket"],
        target_n=180,
    )
    sample_for_personas = sample_for_personas.sort_values(["platform_name", "timestamp"]).reset_index(drop=True)
    sample_for_personas.to_csv(EDA_DIR / "sample_comments_for_personas.csv", index=False)
    sample_for_personas.to_csv(PERSONA_DIR / "sample_comments_for_personas.csv", index=False)

    save_plot_length_histogram(df)
    save_plot_platform_counts(platform_counts)
    for platform_name in monthly_counts["platform_name"].drop_duplicates():
        save_plot_monthly_counts(monthly_counts, platform_name)

    strict_note = {}
    if STRICT_SUMMARY.exists():
        strict_note = load_csv if False else {}
    summary = {
        "dataset_path": str(dataset_path),
        "total_rows": total_rows,
        "unique_threads": unique_threads,
        "unique_comments": unique_comments,
        "counts_by_platform": {
            row["platform_name"]: int(row["count"]) for _, row in platform_counts.iterrows()
        },
        "counts_by_subreddit": {
            row["subreddit"]: int(row["count"]) for _, row in subreddit_counts.iterrows()
        },
        "counts_by_source_type": {
            row["source_type"]: int(row["count"]) for _, row in source_type_counts.iterrows()
        },
        "date_range": {
            "min_date": df["timestamp"].min().date().isoformat() if total_rows else None,
            "max_date": df["timestamp"].max().date().isoformat() if total_rows else None,
        },
        "word_count_summary": {
            "mean": round(float(df["word_count"].mean()), 2) if total_rows else 0.0,
            "median": round(float(df["word_count"].median()), 2) if total_rows else 0.0,
            "p90": round(float(df["word_count"].quantile(0.9)), 2) if total_rows else 0.0,
            "min": int(df["word_count"].min()) if total_rows else 0,
            "max": int(df["word_count"].max()) if total_rows else 0,
        },
        "method_note": (
            "This EDA is based on the current strict working dataset in this repo snapshot. "
            "The strict subset is preliminary and automatically filtered rather than fully manually validated."
        ),
    }
    save_json(EDA_DIR / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EDA on the strict cleaned price-reaction dataset.")
    parser.add_argument("--input", default=str(STRICT_DATASET), help="Path to strict dataset CSV")
    args = parser.parse_args()
    summary = build_eda_outputs(Path(args.input))
    print(f"Saved EDA outputs to: {EDA_DIR}")
    print(pd.Series(summary["counts_by_platform"]).to_string())
    print(f"Rows: {summary['total_rows']}")
    print(f"Threads: {summary['unique_threads']}")


if __name__ == "__main__":
    main()
