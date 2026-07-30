#!/usr/bin/env python3
"""
classify_threads.py
===================
TDOT relevance classification of Reddit threads using the Gemini **Batch API**
(50% cheaper than realtime). Reads the threads.jsonl produced by
reddit_pipeline.py (and optionally the orphans file) and labels each THREAD
(post + comments rendered as a NESTED reply tree, sent as one request):

    0            -> thread is NOT mainly about TDOT-relevant transportation
    1..9         -> thread MAINLY discusses transportation; digit = category
                    1 MAINTENANCE            5 TRANSPORTATION_OPTIONS
                    2 TRAFFIC_OPERATIONS     6 FREIGHT
                    3 SAFETY                 7 TRAVELER_INFO
                    4 CONSTRUCTION           8 FUNDING_POLICY
                    9 OTHER_TDOT (fits no category 1-8; last-resort bucket --
                      its rate is a taxonomy diagnostic)

Prompt v2 design (after v1 false positives, e.g. a political-rally megathread
labeled MAINTENANCE because of sarcastic pothole jokes): RELATED now requires
transportation to be the MAIN topic / a central substantial part of the
discussion (passing mentions, even sincere ones, are 0); overview-first
judgment; an explicit sincerity/sarcasm rule; a Tennessee geographic anchor;
and the reply tree indented so the model sees conversation context.

Workflow (four small steps, each resumable/re-runnable)
-------------------------------------------------------
  # 0. sanity-check prompt + config on a few threads, realtime (costs ~nothing)
  python classify_threads.py test    --threads out/threads.jsonl -n 5

  # 1. build the batch request files + key index. Requests are split into
  #    CHUNKS of ~8M tokens because the Batch API caps enqueued tokens per
  #    model (10M on Tier 1) -- one job per chunk, run sequentially.
  python classify_threads.py prepare --threads out/threads.jsonl \
                                     --orphans out/threads.orphans.jsonl \
                                     --workdir out/classify

  # 2. submit. --wait submits chunk 1, polls until it finishes, submits
  #    chunk 2, and so on. Without --wait, re-run `submit` after each chunk.
  python classify_threads.py submit  --workdir out/classify --wait

  # 3. check progress any time
  python classify_threads.py status  --workdir out/classify

  # 4. when all chunks SUCCEEDED, download + join -> labels CSV
  python classify_threads.py fetch   --workdir out/classify --output out/labels.csv

NOTE: the Batch API requires a PAID tier -- on the free tier every submit
fails with 429 RESOURCE_EXHAUSTED. Link a billing account in AI Studio
(Settings -> Billing); the upgrade to Tier 1 is instant.

API key (never hardcoded)
-------------------------
The google-genai client automatically reads the environment variable
GEMINI_API_KEY (or GOOGLE_API_KEY). Set it once at the OS level -- see the
project notes -- and this script never sees or stores the key itself.
If neither variable is set, the script falls back to a hidden getpass prompt
(like the old popup) so it still runs anywhere.

Requires: Python 3.10+, `pip install google-genai`
"""

from __future__ import annotations

import argparse
import csv
import getpass
import json
import os
import re
import sys
import time

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
MODEL = "gemini-2.5-flash-lite"     # default; override per-run with prepare --model

# Batch-rate prices ($ per 1M tokens: input, output) and observed/documented
# enqueued-token caps, for cost estimates and chunk-size hints.
MODEL_INFO = {
    "gemini-2.5-flash-lite": {"price": (0.05, 0.20), "cap": 3_000_000},
    "gemini-3.1-flash-lite": {"price": (0.125, 0.75), "cap": 10_000_000},
}

CATEGORIES = {
    "0": "NONE",
    "1": "MAINTENANCE",
    "2": "TRAFFIC_OPERATIONS",
    "3": "SAFETY",
    "4": "CONSTRUCTION",
    "5": "TRANSPORTATION_OPTIONS",
    "6": "FREIGHT",
    "7": "TRAVELER_INFO",
    "8": "FUNDING_POLICY",
    "9": "OTHER_TDOT",
}

# The TDOT prompt, v2. Changes vs v1 (which produced false positives like a
# political-rally megathread labeled MAINTENANCE because of sarcastic pothole
# jokes in the comments):
#   - overview-first: judge what the thread AS A WHOLE is about
#   - sincerity rule: sarcastic/rhetorical mentions do not count
#   - Tennessee geographic anchor (local road numbers and street names)
#   - input is the nested reply tree, so the model sees conversation context
#   - two-digit output: category + extent (main topic vs minor part)
SYSTEM_PROMPT = """CONTEXT: You are analyzing conversation threads from Reddit communities in Tennessee, USA (such as r/Tennessee, r/nashville, r/knoxville, r/chattanooga, r/memphis, r/Clarksville, r/murfreesboro, r/RutherfordCountyTN). TDOT is the Tennessee Department of Transportation. When locals mention numbered roads (40, 24, 65, 75, 240, 440, 840, etc.) they usually mean Tennessee interstates or state routes, and local street names (Poplar, Kingston Pike, etc.) are roads in Tennessee cities.
You will receive ONE thread: the post, then its comments, with replies indented under the comment they respond to.
TASK: First, understand what the thread AS A WHOLE is about. Then decide whether the thread MAINLY discusses content relevant to TDOT. A thread is RELATED only if transportation content is the MAIN TOPIC or a central, substantial part of the discussion -- the post itself sincerely raises it, or a large share of the comments sincerely discuss it. The qualifying content is one of the following, concerning interstates, state highways, or other public roads:
1. MAINTENANCE - pavement condition, potholes, resurfacing, cracks, bumps/smoothness; lane striping and pavement marking visibility; roadside litter, debris, or mowing; drainage and standing water on roads; guardrails; bridge condition or closures; snow/ice removal; rest areas and welcome centers; highway lighting.
2. TRAFFIC_OPERATIONS - congestion, traffic jams, bottlenecks, rush-hour delays; traffic signal timing; crash/incident response and lane blockages; HELP trucks / freeway service patrol; ramp metering or interchange design; travel time between or within cities.
3. SAFETY - dangerous roads or intersections; crash-prone locations; visibility of signs; missing or confusing signage; work-zone safety; distracted driving (phone use), speeding, or aggressive driving as a road safety concern; pedestrian or cyclist safety on roadways.
4. CONSTRUCTION - ongoing or planned road construction projects; widening or new highways; construction delays or detours; complaints about how long projects take; lack of advance notice or public information about roadwork; quality of completed projects.
5. TRANSPORTATION_OPTIONS - public transit availability or quality (bus, rail); sidewalks and walkability tied to roads; bike lanes and greenways as transportation; park-and-ride; carpool/vanpool; intercity bus or passenger rail; airport ground access.
6. FREIGHT - volume or behavior of large commercial trucks on highways; truck traffic concerns; freight movement or truck parking.
7. TRAVELER_INFO - road condition or traffic information services (511, TDOT SmartWay, message boards, highway advisory radio); wanting better information about closures, delays, or construction.
8. FUNDING_POLICY - gas tax, tolls, road funding, transportation spending priorities; opinions on what the state should build or fix; whether tax money is well spent on roads.
9. OTHER_TDOT - use ONLY as a last resort: clearly about public road or transportation infrastructure that TDOT could act on, but NO category 1-8 fits even partially. Examples: electric vehicle charging stations along highways, traffic cameras or automated enforcement, highway noise walls, e-scooters/micromobility on roadways. If any category 1-8 fits even loosely, you MUST prefer that category over 9. Category 9 does NOT loosen the NOT RELATED exclusions below.
SINCERITY RULE (important): a topic counts ONLY when raised sincerely as a real issue, experience, or question. A topic mentioned as a joke, sarcastic aside, rhetorical comparison, metaphor, or a way to dismiss another subject does NOT count. Example: in a thread about a political rally, a comment like "we shouldn't protest anything until the potholes are fixed" is sarcasm about protesting, NOT a road maintenance complaint; such a thread is NOT RELATED.
MINOR MENTIONS: if the thread is mainly about something else and transportation appears only in a few passing comments -- even sincere ones -- mark NOT RELATED. Only threads whose discussion is substantially about transportation qualify.
Mark as NOT RELATED: vehicle purchase, repair, insurance, or maintenance (mechanics, tires, oil changes); gas prices as a consumer cost complaint; driver licensing, tags, registration, DMV visits (not TDOT's responsibility); parking availability or parking tickets; rideshare/taxi price or service complaints (unless about road conditions); general travel, tourism, or "how do I get to X" directions; crime or accidents mentioned only as news with no road/infrastructure angle; weather discussion with no road-condition component.
When genuinely ambiguous, prefer NOT RELATED. If multiple categories match, choose the one most central to the transportation content.
OUTPUT FORMAT: Reply with EXACTLY ONE character and nothing else. Reply 0 if NOT RELATED. Otherwise reply the single digit (1-9) of the most central matching category, using 9 only when no category 1-8 fits."""

# Per-request generation settings.
# - temperature 0.0: deterministic-as-possible classification
# - max_output_tokens small: the answer is one character
# - thinking_budget 0: 2.5-series models bill "thinking" as output tokens;
#   for a lookup-style classification we switch it off entirely.
#   (The `test` command validates this config on live calls BEFORE you
#   submit a 24h batch job, so a config error can't waste a day.)
GEN_CONFIG = {
    "temperature": 0.0,
    "max_output_tokens": 8,
    "thinking_config": {"thinking_budget": 0},
}

MAX_THREAD_CHARS = 12_000   # ~3k tokens; very long threads are truncated
MAX_COMMENT_CHARS = 1_500   # per-comment cap before the thread-level cap

STATE_FILE = "batch_state.json"     # inside --workdir
INDEX_FILE = "key_index.csv"


def parse_label(text: str) -> str:
    """Extract the label digit from a model response.

    Gemini occasionally decorates the answer despite the one-character
    instruction ('**2**', 'Label: 0', '0.', a leading newline...). The
    answer digit is still in there, so instead of trusting the first
    character we take the FIRST digit found anywhere in the response.
    Responses are capped at 8 output tokens, so there is no long prose
    for a stray digit to hide in. Returns '' if no digit at all."""
    m = re.search(r"[0-9]", text)
    return m.group(0) if m else ""

# The Batch API caps how many tokens one model can have ENQUEUED at a time
# (Tier 1: 10M tokens for 2.5 Flash-Lite). A big dataset must therefore be
# split into several jobs run one after another. We chunk the request file at
# prepare time; `submit --wait` walks through the chunks automatically.
CHUNK_TOKEN_LIMIT = 2_000_000       # observed real cap: 3M enqueued for
                                    # 2.5-flash-lite; chars/4 underestimates
                                    # by ~10%, so 2M leaves a safe margin
SYSTEM_PROMPT_TOKENS = 1000         # rough size of the v2 prompt, per request


def _chunk_file(workdir: str, i: int) -> str:
    return os.path.join(workdir, f"batch_requests_{i:03d}.jsonl")


def _load_state(workdir: str) -> dict:
    path = os.path.join(workdir, STATE_FILE)
    if not os.path.exists(path):
        sys.exit(f"No {STATE_FILE} in {workdir} -- run `prepare` first.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_state(workdir: str, state: dict) -> None:
    with open(os.path.join(workdir, STATE_FILE), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


ACTIVE_STATES = ("JOB_STATE_PENDING", "JOB_STATE_RUNNING")


def _refresh(client, state: dict) -> None:
    """Update last known job state for every submitted chunk."""
    for chunk in state["chunks"]:
        if chunk.get("job_name"):
            job = client.batches.get(name=chunk["job_name"])
            chunk["last_state"] = job.state.name
            if job.state.name == "JOB_STATE_SUCCEEDED":
                chunk["result_file"] = job.dest.file_name


# --------------------------------------------------------------------------
# Client (API key comes from the environment, never from code)
# --------------------------------------------------------------------------
def make_client():
    try:
        from google import genai
    except ImportError:
        sys.exit("Missing SDK. Run:  pip install google-genai")

    if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
        # Fallback: hidden interactive prompt (nothing is stored on disk).
        os.environ["GEMINI_API_KEY"] = getpass.getpass(
            "GEMINI_API_KEY not found in environment. Paste key (hidden): ")
    return genai.Client()


# --------------------------------------------------------------------------
# Thread -> request text
# --------------------------------------------------------------------------
def thread_to_text(thread: dict) -> tuple[str, int]:
    """Render one threads.jsonl record (post or orphan group) as a text block
    with the REPLY TREE reconstructed from parent_id: replies are indented
    under the comment they respond to, so the model sees conversation context
    (a sarcastic reply reads as a reply, not a standalone complaint).
    Returns (text, n_comments_included)."""
    lines: list[str] = []
    sub = thread.get("subreddit") or ""
    if sub:
        lines.append(f"SUBREDDIT: r/{sub} (Tennessee, USA)")
    if thread.get("missing_post"):
        lines.append("(Post unavailable; comments only.)")
    else:
        lines.append(f"TITLE: {thread.get('title') or ''}")
        selftext = (thread.get("selftext") or "").strip()
        if selftext and selftext not in ("[deleted]", "[removed]"):
            lines.append(f"POST: {selftext[:3 * MAX_COMMENT_CHARS]}")

    comments = thread.get("comments") or []
    included = 0
    if comments:
        # Rebuild the reply tree: t3_<post> or unknown parent -> top level;
        # t1_<comment-in-thread> -> child of that comment.
        by_id = {c["id"]: {**c, "children": []} for c in comments if c.get("id")}
        roots: list[dict] = []
        for c in by_id.values():
            parent = c.get("parent_id") or ""
            pid = parent[3:] if parent.startswith("t1_") else None
            if pid and pid in by_id and pid != c["id"]:
                by_id[pid]["children"].append(c)
            else:
                roots.append(c)
        ts = lambda c: c.get("created_utc") or 0
        roots.sort(key=ts)
        for c in by_id.values():
            c["children"].sort(key=ts)

        lines.append("COMMENTS (replies are indented under the comment they respond to):")
        truncated = False
        # Depth-first with an explicit stack (deep reply chains are rare but real)
        stack = [(c, 0) for c in reversed(roots)]
        while stack:
            if sum(len(l) + 1 for l in lines) > MAX_THREAD_CHARS:
                truncated = True
                break
            c, depth = stack.pop()
            body = (c.get("body") or "").strip()
            if body and body not in ("[deleted]", "[removed]"):
                indent = "  " * min(depth, 8)
                lines.append(f"{indent}- {body[:MAX_COMMENT_CHARS]}")
                included += 1
            for k in reversed(c["children"]):
                stack.append((k, depth + 1))
        if truncated:
            lines.append("(... remaining comments truncated ...)")
    text = "\n".join(lines)
    return text[:MAX_THREAD_CHARS + 2000], included


def build_request(text: str) -> dict:
    """One Batch-API request body (also reused verbatim by `test`)."""
    return {
        "contents": [{"role": "user", "parts": [{"text": text}]}],
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "generation_config": GEN_CONFIG,
    }


def iter_threads(threads_path: str, orphans_path: str | None):
    """Yield (key, record) for every thread; orphan groups get 'orph_' keys."""
    with open(threads_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                yield f"t_{rec['id']}", rec
    if orphans_path and os.path.exists(orphans_path):
        with open(orphans_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    yield f"orph_{rec['id']}", rec


# --------------------------------------------------------------------------
# Step 1: prepare
# --------------------------------------------------------------------------
def cmd_prepare(args) -> None:
    os.makedirs(args.workdir, exist_ok=True)
    idx_path = os.path.join(args.workdir, INDEX_FILE)
    state_path = os.path.join(args.workdir, STATE_FILE)
    if os.path.exists(state_path):
        old = json.load(open(state_path))
        if any(c.get("job_name") for c in old.get("chunks", [])):
            sys.exit(f"{STATE_FILE} shows jobs were already submitted from this "
                     f"workdir. Use a fresh --workdir (or delete the state file "
                     f"if you really want to start over).")

    only_ids: set[str] | None = None
    if args.ids_file:
        with open(args.ids_file, "r", encoding="utf-8") as f:
            only_ids = {line.split("\t")[0].strip() for line in f if line.strip()}
        print(f"Restricting to {len(only_ids):,} ids from {args.ids_file}")

    chunk_limit = args.chunk_tokens
    chunks: list[dict] = []
    reqf = None
    chunk_tokens = 0
    n = 0
    total_tokens = 0

    def open_new_chunk():
        nonlocal reqf, chunk_tokens
        if reqf:
            reqf.close()
        chunks.append({"file": os.path.basename(_chunk_file(args.workdir, len(chunks) + 1)),
                       "n_requests": 0, "est_tokens": 0,
                       "job_name": None, "last_state": None})
        reqf = open(_chunk_file(args.workdir, len(chunks)), "w", encoding="utf-8")
        chunk_tokens = 0

    with open(idx_path, "w", newline="", encoding="utf-8") as idxf:
        idx = csv.writer(idxf)
        idx.writerow(["key", "id", "kind", "subreddit", "n_comments_included", "chunk"])
        open_new_chunk()
        for key, rec in iter_threads(args.threads, args.orphans):
            if only_ids is not None and str(rec.get("id")) not in only_ids:
                continue
            text, n_comments = thread_to_text(rec)
            if not text.strip():
                continue
            est = SYSTEM_PROMPT_TOKENS + len(text) // 4 + 20
            if chunk_tokens + est > chunk_limit and chunks[-1]["n_requests"] > 0:
                open_new_chunk()
            reqf.write(json.dumps(
                {"key": key, "request": build_request(text)},
                ensure_ascii=False) + "\n")
            chunk_tokens += est
            chunks[-1]["n_requests"] += 1
            chunks[-1]["est_tokens"] = chunk_tokens
            kind = "orphan_group" if rec.get("missing_post") else "thread"
            idx.writerow([key, rec.get("id"), kind,
                          rec.get("subreddit", ""), n_comments, len(chunks)])
            n += 1
            total_tokens += est
    if reqf:
        reqf.close()

    _save_state(args.workdir, {"model": args.model, "chunks": chunks})

    p_in, p_out = MODEL_INFO.get(args.model, MODEL_INFO[MODEL])["price"]
    est_cost = total_tokens / 1e6 * p_in + (n * 2) / 1e6 * p_out   # batch prices
    print(f"Prepared {n:,} requests in {len(chunks)} chunk file(s) "
          f"of <= ~{chunk_limit/1e6:.0f}M tokens each (Batch API enqueued-token cap).")
    for i, c in enumerate(chunks, 1):
        print(f"  chunk {i}: {c['n_requests']:,} requests, "
              f"~{c['est_tokens']/1e6:.1f}M tokens -> {c['file']}")
    print(f"Estimated total batch cost with {args.model}: ~${est_cost:.2f} "
          f"(~{total_tokens/1e6:.1f}M input tokens at ${p_in}/M batch rate)")
    cap = MODEL_INFO.get(args.model, {}).get("cap")
    if cap and args.chunk_tokens < cap // 3:
        print(f"TIP: {args.model}'s enqueued-token cap is ~{cap/1e6:.0f}M; "
              f"you could use --chunk-tokens {int(cap*0.8):,} for fewer, larger chunks.")
    print(f"Key index -> {idx_path}")
    print(f"\nNext:  python classify_threads.py submit --workdir {args.workdir} --wait")
    if len(chunks) > 1:
        print("(Multiple chunks: --wait submits them one after another as each "
              "finishes. Without --wait, run `submit` again after each completes.)")


# --------------------------------------------------------------------------
# Step 0: test (realtime, a handful of threads, validates config + prompt)
# --------------------------------------------------------------------------
def cmd_test(args) -> None:
    client = make_client()
    model = args.model or MODEL       # test-only override for model comparison
    wanted = set(args.id) if args.id else None
    tested = 0
    for key, rec in iter_threads(args.threads, args.orphans):
        if wanted is not None:
            if rec.get("id") not in wanted:
                continue
        elif tested >= args.n:
            break
        text, _ = thread_to_text(rec)
        if not text.strip():
            continue
        req = build_request(text)
        resp = client.models.generate_content(
            model=model,
            contents=req["contents"],
            config={
                "system_instruction": SYSTEM_PROMPT,
                **GEN_CONFIG,
            },
        )
        raw = (resp.text or "").strip()
        label = parse_label(raw)
        category = CATEGORIES.get(label, f"UNEXPECTED({raw!r})")
        title = (rec.get("title") or "(comments-only group)")[:70]
        print(f"[{key}] -> {label} {category}   | {title}")
        tested += 1
    print(f"\nTested {tested} threads in realtime. If the labels look sane "
          f"and no errors occurred, the same config is safe to submit as a batch.")


# --------------------------------------------------------------------------
# Step 2: submit
# --------------------------------------------------------------------------
def _submit_chunk(client, workdir: str, chunk: dict, chunk_no: int,
                  model: str = MODEL) -> None:
    from google.genai import errors as genai_errors

    path = os.path.join(workdir, chunk["file"])
    print(f"Uploading chunk {chunk_no} ({chunk['n_requests']:,} requests, "
          f"~{chunk['est_tokens']/1e6:.1f}M tokens) [{model}]...")
    uploaded = client.files.upload(
        file=path,
        config={"display_name": f"tdot-classification-{chunk_no:03d}",
                "mime_type": "jsonl"},
    )
    print(f"Creating batch job for chunk {chunk_no}...")
    # A 429 here is usually TRANSIENT: the previous job just finished, but its
    # tokens take a while to drain from the enqueued-token counter. So we
    # retry with a pause instead of giving up.
    MAX_TRIES = 20
    RETRY_SECONDS = 60
    job = None
    for attempt in range(1, MAX_TRIES + 1):
        try:
            job = client.batches.create(
                model=model,
                src=uploaded.name,
                config={"display_name": f"tdot-classification-{chunk_no:03d}"},
            )
            break
        except genai_errors.ClientError as e:
            if e.code != 429:
                raise
            if attempt == MAX_TRIES:
                sys.exit(
                    f"\nStill 429 RESOURCE_EXHAUSTED after {MAX_TRIES} retries "
                    f"(~{MAX_TRIES * RETRY_SECONDS // 60} min). This is no "
                    "longer a timing hiccup. Check:\n"
                    "  1. Batch API panel at https://ai.dev/rate-limit -- is "
                    "'Batch enqueued tokens' full? A stuck job may be holding "
                    "quota (cancel it with the `cancel` command).\n"
                    "  2. Billing/tier still active for this key's project.\n"
                    "Then re-run `submit --wait` -- it resumes at this chunk.")
            print(f"  429 (quota still draining from the previous job) -- "
                  f"retry {attempt}/{MAX_TRIES} in {RETRY_SECONDS}s")
            time.sleep(RETRY_SECONDS)
    chunk["job_name"] = job.name
    chunk["last_state"] = job.state.name
    print(f"  job: {job.name} ({job.state.name})")


def _poll_until_done(client, chunk: dict, interval: int = 60) -> None:
    # Polling calls batches.get -- a metadata lookup, NOT a model call.
    # It costs no tokens and does not count against model RPM/TPM limits,
    # so a 30-60s interval is safe; don't go below ~15s out of politeness.
    while chunk["last_state"] in ACTIVE_STATES:
        print(f"  {chunk['last_state']} ... next check in {interval}s")
        time.sleep(interval)
        job = client.batches.get(name=chunk["job_name"])
        chunk["last_state"] = job.state.name
        if job.state.name == "JOB_STATE_SUCCEEDED":
            chunk["result_file"] = job.dest.file_name


def cmd_submit(args) -> None:
    client = make_client()
    state = _load_state(args.workdir)
    model = state.get("model", MODEL)
    _refresh(client, state)
    _save_state(args.workdir, state)

    for i, chunk in enumerate(state["chunks"], 1):
        if chunk["last_state"] in ACTIVE_STATES:
            if not args.wait:
                sys.exit(f"Chunk {i} is still {chunk['last_state']}. The "
                         f"enqueued-token cap means we submit one chunk at a "
                         f"time -- re-run `submit` when it finishes, or use "
                         f"--wait to let the script handle all chunks.")
            print(f"Chunk {i} still running; waiting...")
            _poll_until_done(client, chunk, args.poll_seconds)
            _save_state(args.workdir, state)

        if chunk["job_name"] is None:
            _submit_chunk(client, args.workdir, chunk, i, model)
            _save_state(args.workdir, state)
            if not args.wait:
                remaining = sum(1 for c in state["chunks"] if c["job_name"] is None)
                if remaining:
                    print(f"\n{remaining} chunk(s) left. Re-run `submit` when "
                          f"this job finishes, or use --wait next time.")
                else:
                    print("\nLast chunk submitted. Check with `status`; then `fetch`.")
                return
            _poll_until_done(client, chunk, args.poll_seconds)
            _save_state(args.workdir, state)
            if chunk["last_state"] != "JOB_STATE_SUCCEEDED":
                sys.exit(f"Chunk {i} ended in {chunk['last_state']} -- fix and "
                         f"re-submit (see `status`).")

    print("\nAll chunks submitted"
          + (" and finished. Run `fetch` to build the labels CSV." if args.wait
             else ". Check `status`."))


# --------------------------------------------------------------------------
# Step 3: status
# --------------------------------------------------------------------------
def cmd_status(args) -> None:
    client = make_client()
    state = _load_state(args.workdir)
    _refresh(client, state)
    _save_state(args.workdir, state)
    all_done = True
    for i, chunk in enumerate(state["chunks"], 1):
        s = chunk["last_state"] or "NOT_SUBMITTED"
        print(f"chunk {i}: {s:24s} {chunk['n_requests']:,} requests "
              f"({chunk.get('job_name') or '-'})")
        if s != "JOB_STATE_SUCCEEDED":
            all_done = False
    print("\nAll chunks succeeded -- run `fetch`." if all_done else
          "\nNot finished yet. `submit --wait` submits/waits on remaining chunks.")


# --------------------------------------------------------------------------
# errors: explain WHY requests came back without a label
# --------------------------------------------------------------------------
def cmd_errors(args) -> None:
    client = make_client()
    state = _load_state(args.workdir)
    reasons: dict[str, int] = {}
    bad: list[tuple[str, str]] = []       # (key, reason)
    seen_keys: set[str] = set()

    for i, chunk in enumerate(state["chunks"], 1):
        if not chunk.get("result_file"):
            print(f"chunk {i}: no result file (state {chunk.get('last_state')}), skipping")
            continue
        content = client.files.download(file=chunk["result_file"]).decode("utf-8")
        for line in content.splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = row.get("key", "")
            seen_keys.add(key)
            reason = None
            try:
                text = row["response"]["candidates"][0]["content"]["parts"][0]["text"]
                if parse_label(text) not in CATEGORIES:
                    reason = f"no digit in output: {text.strip()[:20]!r}"
            except (KeyError, IndexError, TypeError):
                # No text -- find out why, most specific first
                resp = row.get("response") or {}
                if row.get("error"):
                    reason = f"request error: {row['error'].get('message', '?')[:60]}"
                elif (resp.get("promptFeedback") or {}).get("blockReason"):
                    reason = f"prompt blocked: {resp['promptFeedback']['blockReason']}"
                else:
                    cands = resp.get("candidates") or [{}]
                    fr = cands[0].get("finishReason") or cands[0].get("finish_reason")
                    reason = f"finish_reason: {fr}" if fr else "empty response"
            if reason:
                reasons[reason] = reasons.get(reason, 0) + 1
                bad.append((key, reason))

    # keys in the index that never appeared in any result file
    idx_path = os.path.join(args.workdir, INDEX_FILE)
    with open(idx_path, "r", encoding="utf-8", newline="") as f:
        for rec in csv.DictReader(f):
            if rec["key"] not in seen_keys:
                reasons["missing from results"] = reasons.get("missing from results", 0) + 1
                bad.append((rec["key"], "missing from results"))

    print(f"\n{len(bad):,} failed request(s). Breakdown:")
    for reason, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  {n:6,}  {reason}")

    out = os.path.join(args.workdir, "error_ids.txt")
    with open(out, "w", encoding="utf-8") as f:
        for key, reason in bad:
            rid = key.split("_", 1)[1] if "_" in key else key
            f.write(f"{rid}\t{reason}\n")
    print(f"\nFailed ids -> {out}")
    print("Rescue pass:  prepare --ids-file " + out +
          " --workdir <new_dir>  (then submit as usual)")


# --------------------------------------------------------------------------
# jobs: list recent batch jobs on the account (diagnostic)
# --------------------------------------------------------------------------
def cmd_jobs(args) -> None:
    client = make_client()
    print("Batch jobs on this account (newest first):")
    shown = 0
    active_seen = 0
    for job in client.batches.list(config={"page_size": min(args.n, 100)}):
        state = job.state.name
        flag = "  <-- HOLDING QUOTA" if state in ACTIVE_STATES else ""
        print(f"  {job.name}  {state:22s} "
              f"{getattr(job, 'display_name', '') or ''}{flag}")
        if state in ACTIVE_STATES:
            active_seen += 1
        shown += 1
        if shown >= args.n:
            break
    print(f"\n{active_seen} job(s) currently pending/running. Their tokens "
          f"occupy the enqueued-token cap; cancel a stuck one with:\n"
          f"  python classify_threads.py cancel --job batches/xxxx")


# --------------------------------------------------------------------------
# cancel: terminate jobs on Google's side
# --------------------------------------------------------------------------
def cmd_cancel(args) -> None:
    client = make_client()
    if args.job:
        # Cancel one specific job by name (e.g. the probe: batches/xxxx)
        client.batches.cancel(name=args.job)
        print(f"Cancel requested for {args.job}")
        return
    state = _load_state(args.workdir)
    _refresh(client, state)
    n = 0
    for i, chunk in enumerate(state["chunks"], 1):
        if chunk.get("job_name") and chunk["last_state"] in ACTIVE_STATES:
            client.batches.cancel(name=chunk["job_name"])
            chunk["last_state"] = "JOB_STATE_CANCELLED"
            print(f"chunk {i}: cancel requested ({chunk['job_name']})")
            n += 1
    _save_state(args.workdir, state)
    print(f"{n} active job(s) cancelled." if n else
          "No active jobs to cancel (finished jobs cannot be cancelled, "
          "and they cost nothing further).")


# --------------------------------------------------------------------------
# Step 4: fetch
# --------------------------------------------------------------------------
def cmd_fetch(args) -> None:
    client = make_client()
    state = _load_state(args.workdir)
    _refresh(client, state)
    _save_state(args.workdir, state)

    for i, chunk in enumerate(state["chunks"], 1):
        if chunk["last_state"] in ACTIVE_STATES and args.wait:
            print(f"Chunk {i} still running; waiting...")
            _poll_until_done(client, chunk, args.poll_seconds)
            _save_state(args.workdir, state)
        if chunk["last_state"] != "JOB_STATE_SUCCEEDED":
            sys.exit(f"Chunk {i} is {chunk['last_state'] or 'NOT_SUBMITTED'} -- "
                     f"all chunks must succeed before fetch. See `status` / "
                     f"`submit --wait`, or use fetch --wait.")

    # key -> label, merged across all chunk result files
    labels: dict[str, str] = {}
    errors = 0
    for i, chunk in enumerate(state["chunks"], 1):
        print(f"Downloading results for chunk {i} ({chunk['result_file']})...")
        content = client.files.download(file=chunk["result_file"]).decode("utf-8")
        for line in content.splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = row.get("key", "")
            try:
                text = row["response"]["candidates"][0]["content"]["parts"][0]["text"]
                labels[key] = parse_label(text)
            except (KeyError, IndexError, TypeError):
                labels[key] = ""          # per-request error object
                errors += 1

    # Join with the key index -> final CSV
    idx_path = os.path.join(args.workdir, INDEX_FILE)
    n_out = 0
    with open(idx_path, "r", encoding="utf-8", newline="") as fin, \
            open(args.output, "w", newline="", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        writer = csv.writer(fout)
        writer.writerow(["id", "kind", "subreddit", "n_comments_included",
                         "label", "is_related", "category"])
        for rec in reader:
            label = labels.get(rec["key"], "")
            category = CATEGORIES.get(label, "ERROR")
            is_related = ("True" if label != "" and label in "123456789" else
                          "False" if label == "0" else "ERROR")
            writer.writerow([rec["id"], rec["kind"], rec["subreddit"],
                             rec["n_comments_included"], label,
                             is_related, category])
            n_out += 1

    related = sum(1 for v in labels.values() if v in set("123456789"))
    n_other = sum(1 for v in labels.values() if v == "9")
    print(f"\nWrote {n_out:,} labeled threads -> {args.output}")
    print(f"Related: {related:,} (of which OTHER_TDOT/9: {n_other:,}) | "
          f"Not related: {sum(1 for v in labels.values() if v == '0'):,} | "
          f"Errors/empty: {errors + sum(1 for v in labels.values() if v not in set('0123456789')):,}")
    if related and n_other / max(related, 1) > 0.10:
        print("NOTE: >10% of related threads landed in OTHER_TDOT (9). "
              "Inspect a sample -- the 1-8 taxonomy may be missing a topic.")
    print("(Results stay downloadable from Google for ~6 weeks; the local "
          "CSV is now your permanent copy.)")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="TDOT relevance classification via Gemini Batch API")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("test", help="realtime sanity check on a few threads")
    p.add_argument("--threads", required=True)
    p.add_argument("--orphans", default=None)
    p.add_argument("-n", type=int, default=5)
    p.add_argument("--id", nargs="+", default=None,
                   help="test these specific post ids instead of the first n")
    p.add_argument("--model", default=None,
                   help=f"override model for this test only (default {MODEL}); "
                        "e.g. gemini-2.0-flash-lite to compare quality")

    p = sub.add_parser("prepare", help="build chunked batch request files + key index")
    p.add_argument("--threads", required=True)
    p.add_argument("--orphans", default=None)
    p.add_argument("--workdir", required=True)
    p.add_argument("--chunk-tokens", type=int, default=CHUNK_TOKEN_LIMIT,
                   help="max estimated tokens per batch job (default 2M, "
                        "under the observed 3M enqueued-token cap)")
    p.add_argument("--ids-file", default=None,
                   help="only prepare threads whose id is in this file "
                        "(e.g. error_ids.txt from the `errors` command)")
    p.add_argument("--model", default=MODEL,
                   help=f"model for this run, recorded in batch_state.json and "
                        f"used by submit (default {MODEL})")

    p = sub.add_parser("submit", help="upload + create batch job(s), one chunk at a time")
    p.add_argument("--workdir", required=True)
    p.add_argument("--wait", action="store_true",
                   help="poll each job and auto-submit the next chunk until all done")
    p.add_argument("--poll-seconds", type=int, default=60,
                   help="seconds between status checks (metadata call, costs "
                        "nothing; keep >=15)")

    p = sub.add_parser("status", help="check job state")
    p.add_argument("--workdir", required=True)

    p = sub.add_parser("errors", help="explain why requests failed; write error_ids.txt")
    p.add_argument("--workdir", required=True)

    p = sub.add_parser("jobs", help="list all batch jobs on the account with live states")
    p.add_argument("-n", type=int, default=20, help="max jobs to list")

    p = sub.add_parser("cancel", help="terminate still-running jobs at Google")
    p.add_argument("--workdir", default=None,
                   help="cancel all active jobs recorded in this workdir")
    p.add_argument("--job", default=None,
                   help="cancel one specific job by name (batches/...)")

    p = sub.add_parser("fetch", help="download results -> labels CSV")
    p.add_argument("--workdir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--wait", action="store_true",
                   help="poll until all jobs finish")
    p.add_argument("--poll-seconds", type=int, default=60,
                   help="seconds between status checks")

    args = parser.parse_args()
    if args.command == "cancel" and not args.workdir and not args.job:
        sys.exit("cancel: provide --workdir (cancel all active jobs there) "
                 "or --job batches/xxxx (cancel one job).")
    {"test": cmd_test, "prepare": cmd_prepare, "submit": cmd_submit,
     "status": cmd_status, "jobs": cmd_jobs, "cancel": cmd_cancel,
     "errors": cmd_errors, "fetch": cmd_fetch}[args.command](args)


if __name__ == "__main__":
    main()