# LLM Social Simulation of Streaming-Service Price Hikes

An ongoing University of Queensland capstone that examines whether a small open-source language model can generate Reddit-style reactions to Netflix and Disney+ price changes, and whether those generated reactions resemble real discussion at the population level.

## Project snapshot

- 3,040 records across 27 Reddit threads (3,013 comments and 27 post bodies) were assembled for the current strict working dataset.
- The dataset covers Netflix (2,005 records) and Disney+ (1,035 records) discussions from 25 July 2024 to 25 March 2026.
- The current Qwen2.5-0.5B-Instruct trial generated 200 synthetic comments. Its BERTScore results were Precision 0.8470, Recall 0.8384, and F1 0.8425.
- Aggregate reaction distributions differed between the real and generated samples. BERTScore alone should not be interpreted as evidence of social-simulation fidelity.

## Repository contents

- `src/`: data-collection, filtering, prompt-building, simulation, and evaluation scripts.
- `prompts/`: structured prompt templates without source-comment text.
- `results/`: aggregated tables and figures from the recorded trial.
- `data/`: documentation only; no Reddit text or identifiers are distributed.

## Setup

Create a virtual environment, install `requirements.txt`, and provide a separately obtained, permitted input dataset locally. No API credentials, raw Reddit text, account names, comment identifiers, thread titles, URLs, or generated comment text are included in this repository.

## Interpretation and limitations

This is work in progress. The strict subset was automatically reconstructed and has not been fully manually validated. The recorded MAUVE comparison was skipped because the optional dependency was unavailable in the lightweight evaluation environment. Results are exploratory rather than a claim that the model reproduces real-world opinion.

## Author

Neha Elsa Renji — Master of Data Science student, The University of Queensland.
