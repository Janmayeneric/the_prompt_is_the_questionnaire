# The Prompt Is the Questionnaire

Code and measurement instrument for the paper *"The Prompt Is the Questionnaire:
Reproducible, Survey-Anchored Language Model Measurement of Unsolicited Public
Feedback for a State DOT"* (submitted to the Transportation Research Board
Annual Meeting, 2026). The study reads 109,087 Reddit threads from eight
Tennessee communities with a two-stage large language model workflow whose
prompts are built from the Tennessee DOT's own 2013 and 2016 customer surveys.

The full prompt wordings referenced in the paper live in this repository, inside
the two scripts, as the `SYSTEM_PROMPT` constants.

## What is in this repository

| File | What it is |
|---|---|
| `classify_threads.py` | Stage 1. Screens every thread for relevance to a TDOT service topic. Contains the classification instrument (`SYSTEM_PROMPT`) and the category list. One call per thread; the model answers with a single digit: 0 (not TDOT-relevant) or 1–9 (a category). |
| `extract_issues.py` | Stage 2. Extracts structured issue records from the relevant threads. Contains the extraction instrument (`SYSTEM_PROMPT`), the survey-derived problem-type vocabulary (`ISSUE_VOCAB`, 56 types + OTHER), and the community-to-metro anchors (`SUBREDDIT_METRO`). Up to three records per thread: problem type, named program, stance, roads, scope, place, and a verbatim evidence quote. |
| `fileStreams.py` | Helper for reading the Arctic Shift archive dump files that the thread dataset was assembled from. |

## The measurement configuration

Both stages run with deterministic decoding (temperature 0) against the Gemini
Batch API. The paper's production runs used `gemini-2.5-flash-lite` (the `MODEL`
constant); the cross-model comparison ran `gemini-3.1-flash-lite` through the
same scripts via `--model`. Under this configuration, independent reruns of both
stages returned identical output in our study; this was verified by comparison,
not assumed from the settings.

## How the pipeline runs

Both scripts share the same batch workflow. Set the API key first
(`GEMINI_API_KEY` or `GOOGLE_API_KEY`; if unset, the script asks interactively,
which will stall an unattended run).

```
# Stage 1: classification (input: threads.jsonl)
python classify_threads.py test      --workdir run1   # sanity check on a few threads
python classify_threads.py prepare   --workdir run1
python classify_threads.py submit    --workdir run1 [--wait]
python classify_threads.py status    --workdir run1
python classify_threads.py errors    --workdir run1   # writes error_ids.txt for rescue runs
python classify_threads.py fetch     --workdir run1   # writes labels.csv

# Stage 2: issue extraction (input: threads_related.jsonl, built from labels.csv)
python extract_issues.py  prepare/submit/status/errors/fetch --workdir run2
# fetch writes issues.csv (the fact table) and issues_raw.jsonl
```

Every new run should use a fresh `--workdir`; the `batch_state.json` inside a
workdir tracks that run's jobs and should never be reused.

## Validation gates

Stage 2 checks every record on arrival and flags failures instead of dropping
them (`evidence_ok`, `vocab_ok`, `stance_ok` columns in `issues.csv`):

1. the evidence quote must appear verbatim in the thread text (the
   anti-hallucination gate, checked character by character);
2. the problem type must be on the 56-item list or OTHER;
3. the stance must be one of: negative, positive, mixed, neutral-informational;
4. road references are normalized to canonical forms (I-40, US-70, SR-840).

## Data availability

Reddit thread content is not redistributed here, in keeping with platform
terms. Threads were collected from the Arctic Shift public archive
(https://github.com/ArthurHeitmann/arctic_shift) for eight Tennessee
subreddits, January 2024 through December 2025. The thread ID lists needed to
rebuild the dataset from the archive are available from the authors on request.

## Notes for reuse by other agencies

The method transfers by replacing, not reusing, the instrument: derive the
category list and problem-type vocabulary from your own agency's customer
survey, keep the thread-level unit and the validation gates, and keep the
decoding deterministic. Model versions are retired by vendors over time; the
scripts take the model as a parameter so the instrument outlives any one model.

## Citation

Until the paper is published, please cite this repository:

> Hu, Xinyu. 2026. *The Prompt Is the Questionnaire* (code repository).
> https://github.com/Janmayeneric/the_prompt_is_the_questionnaire
