from __future__ import annotations

import argparse

import pandas as pd

from capstone_pipeline_utils import (
    BROAD_DATASET,
    OFFTOPIC_TITLE_TERMS,
    PLATFORM_NAMES,
    PRICE_TERMS,
    PROCESSED_DIR,
    REACTION_TERMS,
    STRICT_DATASET,
    STRICT_MANIFEST,
    STRICT_SUMMARY,
    STRONG_PRICE_TITLE_TERMS,
    THREAD_REVIEW,
    contains_any,
    deduplicate_comments,
    ensure_output_dirs,
    load_csv,
    parse_timestamp,
    save_json,
)

PLATFORM_ROW_TARGETS = {"Netflix": 2048, "DisneyPlus": 969}
TARGET_THREADS = 27


def score_thread(row: pd.Series) -> tuple[int, dict[str, int]]:
    title = str(row.get("thread_title", "") or "")
    query = str(row.get("queries", "") or "")
    subreddit = str(row.get("subreddit", "") or "")
    platform_name = str(row.get("platform_name", "") or "")

    metrics = {
        "has_strong_price_title": int(contains_any(title, STRONG_PRICE_TITLE_TERMS)),
        "has_price_title": int(contains_any(title, PRICE_TERMS)),
        "has_price_query": int(contains_any(query, PRICE_TERMS)),
        "has_platform_in_title": int(contains_any(title, PLATFORM_NAMES.get(platform_name, []))),
        "has_reaction_title": int(contains_any(title, REACTION_TERMS)),
        "is_offtopic_title": int(contains_any(title, OFFTOPIC_TITLE_TERMS)),
        "has_low_precision_signal": int(
            contains_any(
                title,
                [
                    "fox one",
                    "directv",
                    "jimmy kimmel",
                    "gift card",
                    "what should i watch",
                    "nfl cost",
                ],
            )
        ),
        "subreddit_platform_match": int(subreddit.lower() in {"netflix", "disneyplus", "cordcutters", "streaming"}),
    }

    score = (
        5 * metrics["has_strong_price_title"]
        + 3 * metrics["has_price_title"]
        + 2 * metrics["has_price_query"]
        + 2 * metrics["has_platform_in_title"]
        + 1 * metrics["has_reaction_title"]
        + 1 * metrics["subreddit_platform_match"]
        - 5 * metrics["is_offtopic_title"]
        - 3 * metrics["has_low_precision_signal"]
    )
    return score, metrics


def pick_threads(review_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in review_df.iterrows():
        score, metrics = score_thread(row)
        payload = row.to_dict()
        payload["strict_score"] = score
        payload.update(metrics)
        rows.append(payload)

    scored = pd.DataFrame(rows)
    scored = scored[scored["auto_label"].eq("likely_keep")].copy()
    scored = scored[scored["strict_score"] >= 5].copy()

    selected_parts = []
    remaining_slots = TARGET_THREADS

    for platform_name, target_rows in PLATFORM_ROW_TARGETS.items():
        platform_df = scored[scored["platform_name"].eq(platform_name)].copy()
        platform_df = platform_df.sort_values(
            by=[
                "strict_score",
                "has_strong_price_title",
                "has_price_title",
                "has_platform_in_title",
                "n_rows",
            ],
            ascending=[False, False, False, False, False],
        )

        running_rows = 0
        chosen_indices: list[int] = []
        for idx, row in platform_df.iterrows():
            gap = target_rows - running_rows
            overshoot = running_rows + int(row["n_rows"]) - target_rows

            if gap > 0 and (overshoot <= max(120, gap // 2) or len(chosen_indices) < 6):
                chosen_indices.append(idx)
                running_rows += int(row["n_rows"])

            if running_rows >= target_rows and len(chosen_indices) >= 6:
                break

        chosen = platform_df.loc[chosen_indices].copy()
        selected_parts.append(chosen)
        remaining_slots -= len(chosen)

    selected = pd.concat(selected_parts, ignore_index=True)

    if remaining_slots > 0:
        extra_pool = scored[~scored["thread_id"].isin(selected["thread_id"])].copy()
        extra_pool = extra_pool.sort_values(
            by=[
                "strict_score",
                "has_platform_in_title",
                "has_strong_price_title",
                "n_rows",
            ],
            ascending=[False, False, False, False],
        )
        selected = pd.concat([selected, extra_pool.head(remaining_slots)], ignore_index=True)

    selected = selected.sort_values(
        by=["platform_name", "strict_score", "n_rows"],
        ascending=[True, False, False],
    ).head(TARGET_THREADS)

    return selected.reset_index(drop=True)


def build_strict_dataset() -> dict:
    ensure_output_dirs()

    review_df = load_csv(THREAD_REVIEW)
    combined_df = deduplicate_comments(load_csv(BROAD_DATASET))
    combined_df["timestamp"] = parse_timestamp(combined_df["timestamp"])
    combined_df["text"] = combined_df["text"].fillna("").astype(str)

    selected_threads = pick_threads(review_df)
    strict_df = combined_df[combined_df["thread_id"].isin(selected_threads["thread_id"])].copy()
    strict_df = strict_df.sort_values(["platform_name", "timestamp", "thread_id", "comment_id"]).reset_index(drop=True)

    strict_df.to_csv(STRICT_DATASET, index=False)
    selected_threads.to_csv(STRICT_MANIFEST, index=False)

    summary = {
        "dataset_path": str(STRICT_DATASET),
        "selection_manifest_path": str(STRICT_MANIFEST),
        "selection_mode": "reconstructed_from_thread_review_artifacts",
        "method_note": (
            "This strict dataset was reconstructed from the broad combined Reddit collection and the "
            "autolabeled thread-review file because the strict working CSV was missing in this repo snapshot. "
            "It is a preliminary automatically filtered subset, not a fully manually validated corpus."
        ),
        "total_rows": int(len(strict_df)),
        "unique_threads": int(strict_df["thread_id"].nunique()),
        "platform_counts": {
            key: int(value) for key, value in strict_df.groupby("platform_name").size().to_dict().items()
        },
        "date_range": {
            "min_timestamp": strict_df["timestamp"].min().isoformat() if not strict_df.empty else None,
            "max_timestamp": strict_df["timestamp"].max().isoformat() if not strict_df.empty else None,
        },
        "selected_thread_rows": int(selected_threads["n_rows"].sum()),
        "selected_threads": selected_threads[
            ["platform_name", "thread_id", "thread_title", "n_rows", "strict_score"]
        ].to_dict(orient="records"),
    }
    save_json(STRICT_SUMMARY, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct the strict cleaned price-reaction dataset.")
    parser.parse_args()
    summary = build_strict_dataset()
    print(f"Saved strict dataset: {STRICT_DATASET}")
    print(f"Saved strict summary: {STRICT_SUMMARY}")
    print(pd.Series(summary["platform_counts"], name="rows_by_platform").to_string())
    print(f"Total rows: {summary['total_rows']}")
    print(f"Unique threads: {summary['unique_threads']}")


if __name__ == "__main__":
    main()
