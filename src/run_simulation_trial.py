from __future__ import annotations

import argparse
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from capstone_pipeline_utils import (
    PERSONA_DIR,
    PROMPTS_DIR,
    SIM_DIR,
    STRICT_DATASET,
    dominant_persona_label,
    ensure_output_dirs,
    load_csv,
    parse_timestamp,
    reaction_label,
    save_json,
)

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
FALLBACK_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SYSTEM_PROMPT = (
    "You write one short realistic Reddit comment. "
    "Sound like a regular user, not customer support, a journalist, or a moderator. "
    "Write only the comment itself, with no role labels, no explanations, and no prompt restatement. "
    "Keep it to one short comment, roughly 6 to 24 words. "
    "Do not use greetings, apologies, hashtags, emojis, or support-style language."
)
BANNED_PHRASES = [
    "i'm sorry to hear",
    "i am sorry to hear",
    "please feel free",
    "let me know",
    "our community",
    "hey there",
    "hello there",
    "how can i help",
    "thank you for",
    "i understand",
    "what do you think",
    "let's keep",
]
PROMPT_ECHO_FRAGMENTS = [
    "assistant",
    "system",
    "user",
    "only the comment text",
    "return only the comment text",
    "you are writing one reddit comment",
    "context:",
    "constraints:",
    "persona framing:",
    "tone:",
    "style target:",
    "keep it plausible for a real reddit user",
    "plausible for a real reddit user",
    "write a plausible everyday reaction",
    "return only one short comment",
]
REACTION_GUIDANCE = {
    "neutral_or_information": "Style target: a neutral, matter-of-fact, low-drama reaction.",
    "cancel_or_leave": "Style target: mention leaving, cancelling, or refusing the price if that feels natural.",
    "ad_tier_or_bundle_concern": "Style target: focus on ads, tiers, bundles, or plan structure rather than broad ranting.",
    "value_comparison": "Style target: compare price or value to alternatives in a compact everyday way.",
    "frustrated_but_still_engaged": "Style target: mildly annoyed or worn down, but not a dramatic rant.",
}


def import_generation_backend():
    try:
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        return torch, AutoTokenizer, AutoModelForCausalLM, None
    except Exception as exc:  # pragma: no cover - environment dependent
        return None, None, None, exc


def load_prompt_payload() -> dict:
    path = PROMPTS_DIR / "simulation_prompt_templates.json"
    return json.loads(path.read_text(encoding="utf-8"))


def build_scenarios(real_df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    random.seed(seed)
    working = real_df.copy()
    working["timestamp"] = parse_timestamp(working["timestamp"])
    working["persona_candidate"] = working["text"].map(dominant_persona_label)
    working["reaction_label"] = working["text"].map(reaction_label)
    working = working[working["text"].str.strip().ne("")].copy()

    if "persona_sample_160.csv" in {path.name for path in PERSONA_DIR.glob("*.csv")}:
        sample_pool = load_csv(PERSONA_DIR / "persona_sample_160.csv")
    else:
        sample_pool = pd.DataFrame()

    sample_pool = sample_pool.copy()
    if sample_pool.empty:
        sample_pool = working.sample(n=min(max(n, 160), len(working)), random_state=seed).copy()
    elif len(sample_pool) < n:
        dedup_cols = ["platform_name", "subreddit", "thread_id", "comment_id"]
        existing_keys = set(
            tuple(row) for row in sample_pool[dedup_cols].fillna("").astype(str).itertuples(index=False, name=None)
        )
        working_keys = working[dedup_cols].fillna("").astype(str).apply(tuple, axis=1)
        extra_pool = working.loc[~working_keys.isin(existing_keys)].copy()
        needed = min(n - len(sample_pool), len(extra_pool))
        if needed > 0:
            extra_rows = extra_pool.sample(n=needed, random_state=seed).copy()
            sample_pool = pd.concat([sample_pool, extra_rows], ignore_index=True)

    sample_pool["persona_candidate"] = sample_pool["text"].map(dominant_persona_label)
    sample_pool["reaction_label"] = sample_pool["text"].map(reaction_label)
    sample_pool["scenario_summary"] = sample_pool.apply(
        lambda row: (
            f"A Reddit user is reacting to a {row['platform_name']} subscription price change in r/{row['subreddit']}."
        ),
        axis=1,
    )

    scenarios = sample_pool.sample(n=min(n, len(sample_pool)), random_state=seed).reset_index(drop=True)
    overall_weights = working["reaction_label"].value_counts(normalize=True).to_dict()
    platform_persona_weights = {
        key: group["reaction_label"].value_counts(normalize=True).to_dict()
        for key, group in working.groupby(["platform_name", "persona_candidate"], dropna=False)
    }

    def pick_target_reaction(row: pd.Series) -> str:
        weights = platform_persona_weights.get((row["platform_name"], row["persona_candidate"]), overall_weights)
        labels = list(weights.keys())
        probs = list(weights.values())
        return random.choices(labels, weights=probs, k=1)[0]

    scenarios["target_reaction_label"] = scenarios.apply(pick_target_reaction, axis=1)
    scenarios["scenario_id"] = [f"scenario_{idx:03d}" for idx in range(1, len(scenarios) + 1)]
    return scenarios


def render_prompt(
    template_payload: dict,
    platform_name: str,
    persona_name: str,
    scenario_summary: str,
    target_reaction_label: str,
) -> str:
    base = template_payload["base_template"]["template"]
    persona_lookup = {
        item["persona"]: item["template_addition"] for item in template_payload["persona_templates"]
    }
    scenario_lookup = {
        item["platform"]: item["scenario_summary"] for item in template_payload["platform_scenarios"]
    }

    prompt = base.format(
        platform=platform_name,
        scenario_summary=scenario_summary or scenario_lookup.get(platform_name, scenario_summary),
    )
    prompt += "\n" + persona_lookup.get(persona_name, "Persona framing: Write a plausible everyday reaction.")
    prompt += "\n" + REACTION_GUIDANCE.get(
        target_reaction_label,
        "Style target: keep it brief, plausible, and conversational.",
    )
    prompt += "\nReturn only one short comment."
    return prompt


def clean_generated_comment(text: str) -> str:
    text = str(text or "")
    text = text.replace("\\n", "\n").replace("\r", "\n")
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"(?is)^.*?\bassistant\b\s*[:\-]?\s*", "", text)
    text = re.sub(r"(?im)^\s*(assistant|system|user)\s*[:\-]?\s*", "", text)
    text = re.sub(r"(?is)^.*?\breturn only(?: one short)? comment text?\b\s*[:\-]?\s*", "", text)
    text = re.sub(r"(?is)^.*?\breturn only one short comment\b\s*[:\-]?\s*", "", text)
    text = re.sub(r"(?is)^.*?\bpersona framing\b\s*:\s*", "", text)
    text = re.sub(r"(?is)^.*?\bstyle target\b\s*:\s*", "", text)

    cleaned_lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip(" \t`\"'*-:>.,")
        if not line:
            continue
        lower = line.lower()
        if any(fragment in lower for fragment in PROMPT_ECHO_FRAGMENTS):
            continue
        if lower in {"text", "text.", "comment", "comment.", "assistant"}:
            continue
        cleaned_lines.append(line)

    text = " ".join(cleaned_lines)
    lower = text.lower()
    for fragment in PROMPT_ECHO_FRAGMENTS:
        if fragment in lower:
            text = re.sub(re.escape(fragment), " ", text, flags=re.IGNORECASE)
            lower = text.lower()

    text = re.sub(r"#\w+", " ", text)
    text = re.sub(r"\b(comment|reply)\s*[:\-]\s*", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"(?i)\bassistant\b\s*[:\-]?\s*", " ", text)
    text = re.sub(r"(?i)\b(system|user)\b\s*[:\-]?\s*", " ", text)
    text = re.sub(r"(?i)\bonly the comment text\b", " ", text)
    text = re.sub(r"(?i)\breturn only(?: one short)? comment\b", " ", text)
    text = re.sub(r"(?i)\bkeep it plausible for a real reddit user\b", " ", text)
    text = re.sub(r"(?i)\bwrite a plausible everyday reaction\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" \t`\"'*-:;,.!?")
    if not text:
        return ""

    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept = []
    for sentence in sentences:
        piece = sentence.strip(" \t`\"'*-:;,.")
        if not piece:
            continue
        lower_piece = piece.lower()
        if any(fragment in lower_piece for fragment in PROMPT_ECHO_FRAGMENTS):
            continue
        kept.append(piece)
        if len(kept) >= 2:
            break

    text = " ".join(kept) if kept else text
    text = re.sub(r"\s+", " ", text).strip(" \t`\"'*-:;,.")
    text = re.sub(r"^[^A-Za-z0-9]+", "", text)
    text = re.sub(r"[^A-Za-z0-9.!?']+$", "", text)
    words = text.split()
    if len(words) > 26:
        text = " ".join(words[:26]).rstrip(",;:-")
    return text


def looks_like_reddit_comment(text: str) -> bool:
    if not text:
        return False
    words = text.split()
    if len(words) < 4 or len(words) > 28:
        return False
    lower = text.lower()
    if any(phrase in lower for phrase in BANNED_PHRASES):
        return False
    if any(fragment in lower for fragment in PROMPT_ECHO_FRAGMENTS):
        return False
    if lower.startswith(("hi ", "hey ", "hello ", "assistant ", "system ", "user ")):
        return False
    if "?" in text and len(words) > 14:
        return False
    if sum(text.count(mark) for mark in ".!?") > 2:
        return False
    if re.search(r"\b(we'?ll|you should|i can help|here'?s|i would recommend|i suggest)\b", lower):
        return False
    if re.search(r"\b(thank you|feel free|support|customer service|help you)\b", lower):
        return False
    return True


def generate_with_transformers(
    prompts: list[str],
    model_name: str,
    max_new_tokens: int,
    temperature: float,
    seed: int,
    batch_size: int = 32,
) -> list[str]:
    torch, AutoTokenizer, AutoModelForCausalLM, exc = import_generation_backend()
    if exc is not None:
        raise RuntimeError(
            "Transformers generation backend is unavailable. Install torch and transformers to run generation."
        ) from exc

    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    outputs: list[str] = []

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        batch_messages = [
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            for prompt in batch_prompts
        ]
        input_texts = [
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            for messages in batch_messages
        ]
        inputs = tokenizer(input_texts, return_tensors="pt", padding=True)
        generated = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )

        input_lengths = inputs["attention_mask"].sum(dim=1).tolist()
        batch_outputs: list[str] = []
        for idx, input_len in enumerate(input_lengths):
            completion = tokenizer.decode(
                generated[idx][input_len:],
                skip_special_tokens=True,
            ).strip()
            cleaned = clean_generated_comment(completion)

            if not looks_like_reddit_comment(cleaned):
                retry_messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": batch_prompts[idx]},
                ]
                retry_input_text = tokenizer.apply_chat_template(
                    retry_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                retry_inputs = tokenizer(retry_input_text, return_tensors="pt")
                retry_generated = model.generate(
                    **retry_inputs,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.9,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                )
                retry_completion = tokenizer.decode(
                    retry_generated[0][retry_inputs["input_ids"].shape[-1] :],
                    skip_special_tokens=True,
                ).strip()
                retry_cleaned = clean_generated_comment(retry_completion)
                if looks_like_reddit_comment(retry_cleaned):
                    cleaned = retry_cleaned
                else:
                    cleaned = retry_cleaned or cleaned

            batch_outputs.append(cleaned)

        outputs.extend(batch_outputs)
    return outputs


def run_trial(model_name: str, n: int, seed: int, allow_prompt_only: bool) -> dict:
    ensure_output_dirs()
    real_df = load_csv(STRICT_DATASET)
    template_payload = load_prompt_payload()
    scenarios = build_scenarios(real_df, n=n, seed=seed)
    scenarios["prompt_version"] = "v2"
    scenarios["prompt_text"] = scenarios.apply(
        lambda row: render_prompt(
            template_payload,
            platform_name=row["platform_name"],
            persona_name=row["persona_candidate"],
            scenario_summary=row["scenario_summary"],
            target_reaction_label=row["target_reaction_label"],
        ),
        axis=1,
    )

    outputs_path = SIM_DIR / "synthetic_comments.csv"
    prompts_path = SIM_DIR / "generation_prompts.csv"
    scenarios.to_csv(prompts_path, index=False)

    metadata = {
        "trial_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "fallback_model_name": FALLBACK_MODEL,
        "num_requested_generations": n,
        "num_scenarios": int(len(scenarios)),
        "seed": seed,
        "prompt_version": "v2",
    }

    last_exc: Exception | None = None
    resolved_model_name = model_name
    for candidate_model in [model_name, FALLBACK_MODEL]:
        try:
            generated = generate_with_transformers(
                scenarios["prompt_text"].tolist(),
                model_name=candidate_model,
                max_new_tokens=24,
                temperature=0.45,
                seed=seed,
            )
            resolved_model_name = candidate_model
            break
        except Exception as exc:
            last_exc = exc
            generated = []

    if generated:
        scenarios["generated_comment"] = generated
        scenarios["generation_status"] = "generated"
        metadata["generation_backend"] = "transformers"
        metadata["status"] = "success"
        metadata["resolved_model_name"] = resolved_model_name
        if resolved_model_name != model_name:
            metadata["fallback_used"] = True
    else:
        exc = last_exc or RuntimeError("Unknown generation failure.")
        scenarios["generated_comment"] = ""
        scenarios["generation_status"] = "prompt_only"
        metadata["generation_backend"] = "unavailable"
        metadata["status"] = "blocked_missing_local_model_backend"
        metadata["blocked_reason"] = str(exc)
        if not allow_prompt_only:
            metadata["next_step"] = (
                "Install torch and transformers, or run this script again once the Qwen model is available locally."
            )

    scenarios["model_name"] = model_name
    scenarios["seed"] = seed
    scenarios["generated_at_utc"] = metadata["trial_timestamp_utc"]
    scenarios.to_csv(outputs_path, index=False)
    save_json(SIM_DIR / "trial_metadata.json", metadata)

    with (SIM_DIR / "synthetic_comments.json").open("w", encoding="utf-8") as handle:
        json.dump(scenarios.to_dict(orient="records"), handle, indent=2, ensure_ascii=True, default=str)
        handle.write("\n")

    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the first lightweight simulation trial.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model id")
    parser.add_argument("--n", type=int, default=80, help="Number of synthetic comments to target")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--allow-prompt-only",
        action="store_true",
        help="Do not treat missing generation backend as an error condition.",
    )
    args = parser.parse_args()

    metadata = run_trial(
        model_name=args.model,
        n=args.n,
        seed=args.seed,
        allow_prompt_only=args.allow_prompt_only,
    )
    print(f"Saved simulation outputs to: {SIM_DIR}")
    print(pd.Series(metadata).to_string())


if __name__ == "__main__":
    main()
