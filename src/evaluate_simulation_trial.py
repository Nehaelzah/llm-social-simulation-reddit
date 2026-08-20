from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from capstone_pipeline_utils import (
    EVAL_DIR,
    REPORT_TABLES_DIR,
    SIM_DIR,
    STRICT_DATASET,
    configure_matplotlib,
    deduplicate_comments,
    dominant_persona_label,
    ensure_output_dirs,
    generation_quality_note,
    load_csv,
    parse_timestamp,
    reaction_label,
    save_json,
    sentiment_bucket,
    stance_bucket,
    word_count,
)


def try_bertscore(references: list[str], candidates: list[str]) -> dict:
    try:
        from bert_score import score as bert_score  # type: ignore

        precision, recall, f1 = bert_score(candidates, references, lang="en", verbose=False)
        return {
            "status": "computed",
            "precision_mean": round(float(precision.mean().item()), 4),
            "recall_mean": round(float(recall.mean().item()), 4),
            "f1_mean": round(float(f1.mean().item()), 4),
        }
    except Exception as exc:  # pragma: no cover - environment dependent
        return {
            "status": "skipped",
            "reason": str(exc),
        }


def try_mauve(references: list[str], candidates: list[str]) -> dict:
    try:
        import mauve  # type: ignore

        result = mauve.compute_mauve(
            p_text=references,
            q_text=candidates,
            device_id=-1,
            verbose=False,
            max_text_length=128,
        )
        return {
            "status": "computed",
            "mauve": round(float(result.mauve), 4),
        }
    except Exception as exc:  # pragma: no cover - environment dependent
        return {
            "status": "skipped",
            "reason": str(exc),
            "fallback_note": (
                "MAUVE is optional in this local workflow because it can be heavy and brittle on lightweight CPU setups. "
                "Keep it in the methodology, but treat BERTScore plus stance/sentiment distributions as the practical first-pass result."
            ),
        }


def save_wordcount_plot(paired: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    max_count = int(
        max(
            paired["reference_word_count"].max(),
            paired["generated_word_count"].max(),
            10,
        )
    )
    bins = range(0, max_count + 5, 5)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        paired["reference_word_count"],
        bins=bins,
        alpha=0.55,
        density=True,
        color="#1f2937",
        label="Real strict comments",
    )
    ax.hist(
        paired["generated_word_count"],
        bins=bins,
        alpha=0.55,
        density=True,
        color="#0f766e",
        label="Generated comments",
    )
    ax.set_title("Real vs Generated Word Count Distribution")
    ax.set_xlabel("Word count")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(REPORT_TABLES_DIR / "real_vs_generated_wordcount.png", dpi=180)
    plt.close(fig)


def save_reaction_plot(reaction_df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    pivot = (
        reaction_df.pivot(index="reaction_label", columns="set_name", values="share")
        .fillna(0)
        .sort_index()
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", ax=ax, color=["#1f2937", "#0f766e"])
    ax.set_title("Reaction Category Distribution: Real vs Generated")
    ax.set_xlabel("Reaction category")
    ax.set_ylabel("Share")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(frameon=False, title="")
    fig.tight_layout()
    fig.savefig(REPORT_TABLES_DIR / "reaction_distribution_plot.png", dpi=180)
    plt.close(fig)


def build_distribution_table(
    labels: pd.Series,
    set_name: str,
    label_col: str,
) -> pd.DataFrame:
    counts = labels.value_counts(dropna=False).rename_axis(label_col).reset_index(name="count")
    counts["set_name"] = set_name
    counts["share"] = (counts["count"] / counts["count"].sum()).round(4)
    return counts[["set_name", label_col, "count", "share"]]


def build_eval_outputs(sim_path: Path = SIM_DIR / "synthetic_comments.csv") -> dict:
    configure_matplotlib()
    ensure_output_dirs()

    if not sim_path.exists():
        metadata = {
            "status": "skipped",
            "reason": f"Simulation file not found: {sim_path}",
            "next_step": "Run run_simulation_trial.py first so synthetic_comments.csv exists.",
        }
        save_json(EVAL_DIR / "evaluation_summary.json", metadata)
        return metadata

    real_df = deduplicate_comments(load_csv(STRICT_DATASET))
    real_df["timestamp"] = parse_timestamp(real_df["timestamp"])
    real_df["text"] = real_df["text"].fillna("").astype(str)
    real_df = real_df[real_df["text"].str.strip().ne("")].copy()
    real_df["persona_candidate"] = real_df["text"].map(dominant_persona_label)

    synth_df = load_csv(sim_path)
    synth_df["generated_comment"] = synth_df["generated_comment"].fillna("").astype(str)
    synth_df = synth_df[synth_df["generated_comment"].str.strip().ne("")].copy()

    if synth_df.empty:
        metadata = {
            "status": "skipped",
            "reason": "No generated comments were available. Run the simulation trial after installing a local model backend.",
        }
        save_json(EVAL_DIR / "evaluation_summary.json", metadata)
        return metadata

    real_sample = (
        real_df.groupby("platform_name", dropna=False)
        .head(max(1, len(synth_df) // max(1, real_df["platform_name"].nunique())))
        .head(len(synth_df))
        .reset_index(drop=True)
    )
    real_sample = real_sample.head(len(synth_df)).copy()
    synth_df = synth_df.head(len(real_sample)).copy()

    paired = pd.DataFrame(
        {
            "platform_name": synth_df["platform_name"].tolist(),
            "persona_candidate": synth_df["persona_candidate"].tolist(),
            "reference_comment": real_sample["text"].tolist(),
            "generated_comment": synth_df["generated_comment"].tolist(),
        }
    )
    paired["reference_word_count"] = paired["reference_comment"].map(word_count)
    paired["generated_word_count"] = paired["generated_comment"].map(word_count)
    paired["reference_sentiment"] = paired["reference_comment"].map(sentiment_bucket)
    paired["generated_sentiment"] = paired["generated_comment"].map(sentiment_bucket)
    paired["reference_stance"] = paired["reference_comment"].map(stance_bucket)
    paired["generated_stance"] = paired["generated_comment"].map(stance_bucket)
    paired["reference_reaction_label"] = paired["reference_comment"].map(reaction_label)
    paired["generated_reaction_label"] = paired["generated_comment"].map(reaction_label)
    paired["generated_quality_note"] = paired["generated_comment"].map(generation_quality_note)
    paired.to_csv(EVAL_DIR / "paired_reference_generated.csv", index=False)

    bertscore_summary = try_bertscore(
        references=paired["reference_comment"].tolist(),
        candidates=paired["generated_comment"].tolist(),
    )
    mauve_summary = try_mauve(
        references=paired["reference_comment"].tolist(),
        candidates=paired["generated_comment"].tolist(),
    )

    reaction_df = pd.concat(
        [
            build_distribution_table(paired["reference_reaction_label"], "real", "reaction_label"),
            build_distribution_table(paired["generated_reaction_label"], "generated", "reaction_label"),
        ],
        ignore_index=True,
    )
    reaction_df.to_csv(REPORT_TABLES_DIR / "reaction_distribution_comparison.csv", index=False)

    sentiment_df = pd.concat(
        [
            build_distribution_table(paired["reference_sentiment"], "real", "sentiment_label"),
            build_distribution_table(paired["generated_sentiment"], "generated", "sentiment_label"),
        ],
        ignore_index=True,
    )
    sentiment_df.to_csv(REPORT_TABLES_DIR / "sentiment_distribution_comparison.csv", index=False)

    summary_wordcount_rows = [
        {
            "set_name": "real",
            "avg_word_count": round(float(paired["reference_word_count"].mean()), 2),
            "median_word_count": round(float(paired["reference_word_count"].median()), 2),
        },
        {
            "set_name": "generated",
            "avg_word_count": round(float(paired["generated_word_count"].mean()), 2),
            "median_word_count": round(float(paired["generated_word_count"].median()), 2),
        },
    ]
    pd.DataFrame(summary_wordcount_rows).to_csv(EVAL_DIR / "distribution_comparison.csv", index=False)
    save_wordcount_plot(paired)
    save_reaction_plot(reaction_df)

    summary = {
        "status": "completed",
        "num_compared_comments": int(len(paired)),
        "bertscore": bertscore_summary,
        "mauve": mauve_summary,
        "reaction_distribution_path": str(REPORT_TABLES_DIR / "reaction_distribution_comparison.csv"),
        "sentiment_distribution_path": str(REPORT_TABLES_DIR / "sentiment_distribution_comparison.csv"),
        "wordcount_plot_path": str(REPORT_TABLES_DIR / "real_vs_generated_wordcount.png"),
        "reaction_plot_path": str(REPORT_TABLES_DIR / "reaction_distribution_plot.png"),
        "method_note": (
            "BERTScore is the preferred first-pass text similarity metric here. "
            "MAUVE is attempted but may be skipped in lightweight local environments."
        ),
    }
    save_json(EVAL_DIR / "evaluation_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the first simulation trial.")
    parser.add_argument("--input", default=str(SIM_DIR / "synthetic_comments.csv"), help="Simulation CSV path")
    args = parser.parse_args()
    summary = build_eval_outputs(Path(args.input))
    print(f"Saved evaluation outputs to: {EVAL_DIR}")
    print(pd.Series(summary).to_string())


if __name__ == "__main__":
    main()
