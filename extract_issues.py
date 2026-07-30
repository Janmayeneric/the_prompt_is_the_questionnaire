#!/usr/bin/env python3
"""
extract_issues.py
=================
STAGE 2 of the Sentiment Project pipeline: ISSUE EXTRACTION on the threads
already classified as TDOT-related by classify_threads.py (stage 1).

Input:  threads_related.jsonl  (one line per related thread; each record
        already carries "label" (1-9) and "category" from stage 1 — the
        prompt TELLS the model this prior classification and constrains the
        problem_type vocabulary accordingly).
Output: issues.csv — a flat FACT TABLE, one row per extracted issue,
        two-sided (WHAT: problem_type/program | WHERE: roads/scope/place),
        plus stance, reason, and a verbatim evidence quote that is
        programmatically verified against the thread text (anti-hallucination
        gate). Also issues_raw.jsonl for debugging.

Same Gemini Batch API workflow as classify_threads.py (test / prepare /
submit / status / jobs / cancel / errors / fetch), same resumable state
machinery, same operational facts (see claude/batch-run-operational-notes.md:
paid tier required, ~3M real enqueued-token cap for 2.5-flash-lite, 429s
after a job finishes are normal drain lag).

    # 0. sanity check on a few threads, realtime
    python extract_issues.py test    --threads threads_related.jsonl -n 5

    # 1. build chunked batch request files
    python extract_issues.py prepare --threads threads_related.jsonl \
                                     --workdir out/issues

    # 2-3. submit and watch
    python extract_issues.py submit  --workdir out/issues --wait
    python extract_issues.py status  --workdir out/issues

    # 4. download, parse, VERIFY EVIDENCE, write the fact table
    python extract_issues.py fetch   --workdir out/issues \
                                     --threads threads_related.jsonl \
                                     --output issues.csv

Requires: Python 3.10+, `pip install google-genai`
API key: GEMINI_API_KEY / GOOGLE_API_KEY from the environment (never stored).
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
MODEL = "gemini-2.5-flash-lite"

# Subreddit -> metro anchor (the geographic prior; r/Tennessee = statewide).
# Keys are lowercased for lookup.
SUBREDDIT_METRO = {
    "nashville": "Nashville",
    "knoxville": "Knoxville",
    "chattanooga": "Chattanooga",
    "memphis": "Memphis",
    "clarksville": "Clarksville",
    "murfreesboro": "Murfreesboro",
    "rutherfordcountytn": "Murfreesboro / Rutherford County",
    "tennessee": "Tennessee statewide",
}

# Survey-anchored controlled vocabulary (design doc v4.1). Sources: TDOT 2016
# Statewide Customer Survey items (ETC Institute) + stage-1 category
# definitions + expected Reddit-native additions. OTHER is the safety valve;
# its rate is a coverage diagnostic (same logic as OTHER_TDOT in stage 1).
ISSUE_VOCAB: dict[str, list[str]] = {
    "MAINTENANCE": [
        "pavement_condition",            # potholes, cracks, rough surface (Q1.7-8)
        "debris_on_roadway",             # Q1.1
        "litter_and_trash",              # Q1.2
        "snow_ice_removal",              # Q1.3
        "vegetation_mowing",             # Q1.4
        "guardrail_barrier_condition",   # Q1.5
        "drainage_flooding",             # Q1.6
        "shoulder_condition",            # Q1.11
        "bridge_condition",              # Q1.12
        "striping_visibility",           # Q1.13-15
        "signage_condition_clarity",     # Q1.16-17
        "rest_areas",                    # Q1.9-10
        "highway_lighting",
    ],
    "TRAFFIC_OPERATIONS": [
        "recurring_congestion",          # Q1.18-19, Q7.13-14
        "incident_response_clearance",   # HELP trucks; Q1.20-21, Q7.10-11
        "signal_timing",
        "work_zone_traffic_delays",      # Q7.8
        "detour_navigation",             # Q7.4-7
        "interchange_ramp_design",
        "its_technology",                # cameras, message signs, sensors (Q24.3)
    ],
    "SAFETY": [
        "dangerous_road_or_intersection",
        "work_zone_safety",              # Q7.2-3
        "reckless_distracted_driving",
        "pedestrian_cyclist_safety",
        "overall_highway_safety",        # Q7.1
    ],
    "CONSTRUCTION": [
        "construction_delays",           # Q7.8
        "construction_communication",    # Q7.12, Q8c
        "business_access_during_construction",  # Q7.9
        "community_involvement_planning",       # Q8b
        "project_quality_outcome",       # Q8a
        "proposed_project_debate",       # widening / new highway discussions
    ],
    "TRANSPORTATION_OPTIONS": [
        "transit_availability",          # Q5.1
        "transit_frequency_reliability", # Q5.2
        "transit_access_proximity",      # Q5.3
        "accessibility_elderly_disabled",# Q5.4
        "pedestrian_facilities",         # Q5.5
        "bicycle_facilities",            # Q5.6
        "park_and_ride",                 # Q5.7
        "passenger_rail_intercity",      # rail / intercity bus (Reddit-native)
        "airport_ground_access",
    ],
    "FREIGHT": [
        "truck_traffic",
        "truck_parking",
        "freight_movement",
    ],
    "TRAVELER_INFO": [
        "road_condition_information",    # 511, SmartWay, message boards (Q11)
        "digital_services",              # website/apps (Q10)
    ],
    "FUNDING_POLICY": [
        "transportation_funding_taxes",  # gas tax, spending, value (Q21-23)
        "funding_priorities",            # Q24-25
        "toll_choice_lanes",             # Choice Lanes program (post-2016)
        "rural_access_investment",       # Q24.5
    ],
    "OTHER_TDOT": [
        "customer_service_responsiveness",  # Q9
        "environmental_concerns",           # Q17.5
        "agency_trust_performance",         # Q17
        "ev_charging",
        "traffic_cameras_enforcement",
        "noise_walls",
        "micromobility",
    ],
}

ALL_TYPES = {t for lst in ISSUE_VOCAB.values() for t in lst} | {"OTHER"}


def _vocab_block() -> str:
    lines = []
    for cat, types in ISSUE_VOCAB.items():
        lines.append(f"{cat}: " + ", ".join(types))
    return "\n".join(lines)


# One STATIC system prompt (cache-friendly, same pattern as stage 1). The
# thread's stage-1 category arrives in the per-request user text; the model
# is told the classification already happened and is trusted.
SYSTEM_PROMPT = f"""CONTEXT: You are analyzing conversation threads from Reddit communities in Tennessee, USA. TDOT is the Tennessee Department of Transportation. When locals mention numbered roads (40, 24, 65, 75, 240, 440, 840, etc.) they usually mean Tennessee interstates or state routes, and local street names (Poplar, Kingston Pike, etc.) are roads in Tennessee cities.

Each thread you receive was ALREADY CLASSIFIED in an earlier stage as substantially discussing a TDOT-relevant topic, and its assigned CATEGORY is given in the header. Trust that classification: do not re-judge relevance. Your job is the NEXT level of detail: identify the specific issue(s) the thread substantially discusses.

You will receive ONE thread: the post, then its comments, with replies indented under the comment they respond to. Read the WHOLE thread — the real issue is often synthesized from the post plus diagnostic details buried in comments (the post asks "why is 40 stopped every morning?", a comment explains "bridge work at exit 407 until October"; the issue is the synthesis).

TASK: report the issue(s) as structured records with TWO INDEPENDENT SIDES — WHAT the issue is (never geographic) and WHERE it is (only geographic). Report at most 3 issues; most threads have exactly 1. A topic counts only when raised sincerely and substantially (jokes, sarcasm, and passing mentions are NOT issues — same standard as the earlier classification). If, on reading, nothing qualifies, return an empty list.

CONTROLLED VOCABULARY for "problem_type" (grouped by category):
{_vocab_block()}

The primary issue's problem_type normally comes from the thread's assigned category's list. A genuine secondary issue may use a type from any category's list. If no listed type fits an issue even loosely, use exactly "OTHER" and describe it in the gist.

FIELDS per issue:
- "problem_type": one identifier from the vocabulary above, or "OTHER".
- "program_or_policy": named program / policy / agency at the center, in canonical form ("Choice Lanes", "gas tax", "IMPROVE Act", "WeGo bus system", "MATA", "KAT", "TDOT SmartWay"), or null. Roads and highways are NEVER programs — they go in "roads".
- "roads": list of canonical route references substantially discussed, e.g. ["I-40"] or ["I-24", "I-75"]. Canonical forms: "I-40", "US-70", "SR-840", or a named local road ("Kingston Pike", "Sam Cooper Boulevard"). Empty list if none.
- "scope": "state" | "metro" | "corridor" | "point". Default is "metro" (the subreddit's area, given in the header). Use "state" for statewide programs/policies. Use "corridor" or "point" ONLY if the text names a segment, landmark, exit, or intersection. NEVER invent precision the text does not contain.
- "place": short place description ("Knoxville area", "I-24 between exits 60 and 66", "Tennessee statewide").
- "anchor_city": nearest city, or null for state scope.
- "statement": ONE sentence, maximum 12 words, composed from problem_type + program_or_policy + road/place. Describe the CONDITION, not the conversation. GOOD: "Construction delays on I-24 near Murfreesboro". BAD: "People complaining about construction".
- "gist": 1-2 sentences on what the discussion adds (diagnosis, timeline, context, proposed causes).
- "stance": overall community opinion in this thread toward the issue: "negative" | "positive" | "mixed" | "neutral_informational".
- "reason": 3-8 words on why ("delays and lane closures", "wants more transit options").
- "evidence": ONE quote from the thread, maximum 25 words, copied VERBATIM character for character — no paraphrase, no spelling fixes, never merge two comments. Choose the quote that best demonstrates the issue.

RULES:
1. TENNESSEE ANCHOR: bare route numbers and road names are Tennessee roads near the subreddit's area unless the thread clearly says otherwise. A bare "40" counts as I-40 only when context shows a road is meant ("40 was a parking lot" yes; "$40" no).
2. ONE PROBLEM PER ISSUE: distinct problems on the same road = separate issue records; the same problem discussed for several roads = ONE record listing the roads.
3. WHAT/WHERE SEPARATION: problem_type and program_or_policy must contain no place or road names.

OUTPUT: JSON only, exactly this shape, nothing else:
{{"issues": [{{"problem_type": "...", "program_or_policy": null, "roads": [], "scope": "metro", "place": "...", "anchor_city": "...", "statement": "...", "gist": "...", "stance": "...", "reason": "...", "evidence": "..."}}]}}
or {{"issues": []}}"""

# Extraction returns a JSON object, not a digit: JSON mode on, room for up to
# 3 issue records, thinking off (billing: 2.5-series bills thinking as output).
GEN_CONFIG = {
    "temperature": 0.0,
    "max_output_tokens": 1400,
    "response_mime_type": "application/json",
    "thinking_config": {"thinking_budget": 0},
}

MAX_THREAD_CHARS = 12_000
MAX_COMMENT_CHARS = 1_500

STATE_FILE = "batch_state.json"
INDEX_FILE = "key_index.csv"

CHUNK_TOKEN_LIMIT = 2_000_000       # same observed enqueued cap as stage 1
SYSTEM_PROMPT_TOKENS = 1600
OUTPUT_TOKENS_EST = 500             # per-request output estimate for cost


# --------------------------------------------------------------------------
# Parsing & validation
# --------------------------------------------------------------------------
STANCES = {"negative", "positive", "mixed", "neutral_informational"}

_ROAD_PATTERNS = [
    (re.compile(r"^\s*(?:i|interstate)[\s\-]*(\d+)\s*$", re.I), "I-{}"),
    (re.compile(r"^\s*(?:us|u\.s\.|highway|hwy)[\s\-]*(\d+)\s*$", re.I), "US-{}"),
    (re.compile(r"^\s*(?:sr|tn|state route|route)[\s\-]*(\d+)\s*$", re.I), "SR-{}"),
]


def normalize_road(r: str) -> str:
    """Canonicalize obvious route-number variants; leave named roads as-is."""
    r = (r or "").strip()
    for pat, fmt in _ROAD_PATTERNS:
        m = pat.match(r)
        if m:
            return fmt.format(m.group(1))
    return r


_WS = re.compile(r"\s+")


def _norm_text(s: str) -> str:
    """Whitespace-collapsed, casefolded, smart-quote-normalized — the lenient
    form used for the evidence-substring gate."""
    s = (s or "").replace("’", "'").replace("‘", "'")
    s = s.replace("“", '"').replace("”", '"').replace("…", "...")
    return _WS.sub(" ", s).casefold().strip()


def parse_issues(raw: str) -> tuple[list[dict], str]:
    """Parse the model's JSON. Returns (issues, error). Lenient about code
    fences and stray text around the object (strict prompt, lenient parser —
    the stage-1 philosophy)."""
    if not raw or not raw.strip():
        return [], "empty response"
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)     # first {...} block anywhere
        if not m:
            return [], f"no JSON object in output: {text[:40]!r}"
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError as e:
            return [], f"invalid JSON: {e}"
    issues = obj.get("issues")
    if issues is None:
        return [], "JSON lacks 'issues' key"
    if not isinstance(issues, list):
        return [], "'issues' is not a list"
    return issues[:3], ""


def validate_issue(iss: dict, thread_text_norm: str) -> dict:
    """Apply the deterministic gates; returns a flat, csv-ready dict with
    *_ok flags. Never raises — bad fields become flags, not crashes."""
    pt = str(iss.get("problem_type") or "").strip()
    roads = iss.get("roads") or []
    if not isinstance(roads, list):
        roads = [str(roads)]
    roads = [normalize_road(str(r)) for r in roads if str(r).strip()]
    evidence = str(iss.get("evidence") or "").strip().strip('"')
    stance = str(iss.get("stance") or "").strip()
    return {
        "problem_type": pt,
        "vocab_ok": pt in ALL_TYPES,
        "program_or_policy": (iss.get("program_or_policy") or "") or "",
        "roads": "|".join(roads),
        "scope": str(iss.get("scope") or ""),
        "place": str(iss.get("place") or ""),
        "anchor_city": (iss.get("anchor_city") or "") or "",
        "statement": str(iss.get("statement") or ""),
        "gist": str(iss.get("gist") or ""),
        "stance": stance,
        "stance_ok": stance in STANCES,
        "reason": str(iss.get("reason") or ""),
        "evidence": evidence,
        "evidence_ok": bool(evidence) and _norm_text(evidence) in thread_text_norm,
    }


# --------------------------------------------------------------------------
# Shared plumbing (same shape as classify_threads.py)
# --------------------------------------------------------------------------
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
    for chunk in state["chunks"]:
        if chunk.get("job_name"):
            job = client.batches.get(name=chunk["job_name"])
            chunk["last_state"] = job.state.name
            if job.state.name == "JOB_STATE_SUCCEEDED":
                chunk["result_file"] = job.dest.file_name


def make_client():
    try:
        from google import genai
    except ImportError:
        sys.exit("Missing SDK. Run:  pip install google-genai")
    if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
        os.environ["GEMINI_API_KEY"] = getpass.getpass(
            "GEMINI_API_KEY not found in environment. Paste key (hidden): ")
    return genai.Client()


# --------------------------------------------------------------------------
# Thread -> request text (stage-1 renderer + stage-2 header)
# --------------------------------------------------------------------------
def thread_to_text(thread: dict) -> tuple[str, int]:
    """Same nested-tree rendering as classify_threads.py, with a header that
    carries the geographic prior (subreddit metro) and the stage-1 category."""
    lines: list[str] = []
    sub = thread.get("subreddit") or ""
    metro = SUBREDDIT_METRO.get(sub.lower(), "Tennessee")
    if sub:
        lines.append(f"SUBREDDIT: r/{sub} (area: {metro}, Tennessee, USA)")
    cat = thread.get("category") or ""
    if cat:
        lines.append(f"CLASSIFIED CATEGORY (from earlier stage, trusted): {cat}")
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
    return {
        "contents": [{"role": "user", "parts": [{"text": text}]}],
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "generation_config": GEN_CONFIG,
    }


def iter_threads(threads_path: str):
    """threads_related.jsonl: every record already has label/category."""
    with open(threads_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                if rec.get("missing_post"):
                    continue            # orphans were never classified
                yield f"t_{rec['id']}", rec


# --------------------------------------------------------------------------
# test (realtime, with live gate checks)
# --------------------------------------------------------------------------
def cmd_test(args) -> None:
    client = make_client()
    model = args.model or MODEL
    wanted = set(args.id) if args.id else None
    tested = 0
    for key, rec in iter_threads(args.threads):
        if wanted is not None:
            if rec.get("id") not in wanted:
                continue
        elif tested >= args.n:
            break
        text, _ = thread_to_text(rec)
        if not text.strip():
            continue
        resp = client.models.generate_content(
            model=model,
            contents=[{"role": "user", "parts": [{"text": text}]}],
            config={"system_instruction": SYSTEM_PROMPT, **GEN_CONFIG},
        )
        raw = (resp.text or "").strip()
        issues, err = parse_issues(raw)
        title = (rec.get("title") or "")[:70]
        print(f"\n[{key}] {rec.get('category')} | {title}")
        if err:
            print(f"  PARSE ERROR: {err}\n  raw: {raw[:200]!r}")
        norm = _norm_text(text)
        for i, iss in enumerate(issues, 1):
            v = validate_issue(iss, norm)
            flags = []
            if not v["vocab_ok"]:
                flags.append(f"UNKNOWN TYPE {v['problem_type']!r}")
            if not v["evidence_ok"]:
                flags.append("EVIDENCE NOT FOUND IN THREAD")
            if not v["stance_ok"]:
                flags.append(f"BAD STANCE {v['stance']!r}")
            flag = ("  <-- " + "; ".join(flags)) if flags else ""
            print(f"  {i}. [{v['problem_type']}] {v['statement']}{flag}")
            print(f"     where: roads={v['roads'] or '-'} scope={v['scope']} "
                  f"place={v['place']}")
            print(f"     stance={v['stance']} ({v['reason']})")
            print(f"     evidence: {v['evidence'][:90]!r}")
        if not issues and not err:
            print("  (no issues reported)")
        tested += 1
    print(f"\nTested {tested} threads in realtime. If extractions look sane and "
          f"evidence checks pass, prepare/submit the batch.")


# --------------------------------------------------------------------------
# prepare
# --------------------------------------------------------------------------
def cmd_prepare(args) -> None:
    os.makedirs(args.workdir, exist_ok=True)
    idx_path = os.path.join(args.workdir, INDEX_FILE)
    state_path = os.path.join(args.workdir, STATE_FILE)
    if os.path.exists(state_path):
        old = json.load(open(state_path))
        if any(c.get("job_name") for c in old.get("chunks", [])):
            sys.exit(f"{STATE_FILE} shows jobs were already submitted from this "
                     f"workdir. Use a fresh --workdir.")

    only_ids: set[str] | None = None
    if args.ids_file:
        with open(args.ids_file, "r", encoding="utf-8") as f:
            only_ids = {line.split("\t")[0].strip() for line in f if line.strip()}
        print(f"Restricting to {len(only_ids):,} ids from {args.ids_file}")

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
        idx.writerow(["key", "id", "subreddit", "category",
                      "n_comments_included", "chunk"])
        open_new_chunk()
        for key, rec in iter_threads(args.threads):
            if only_ids is not None and str(rec.get("id")) not in only_ids:
                continue
            text, n_comments = thread_to_text(rec)
            if not text.strip():
                continue
            est = SYSTEM_PROMPT_TOKENS + len(text) // 4 + OUTPUT_TOKENS_EST
            if chunk_tokens + est > args.chunk_tokens and chunks[-1]["n_requests"] > 0:
                open_new_chunk()
            reqf.write(json.dumps({"key": key, "request": build_request(text)},
                                  ensure_ascii=False) + "\n")
            chunk_tokens += est
            chunks[-1]["n_requests"] += 1
            chunks[-1]["est_tokens"] = chunk_tokens
            idx.writerow([key, rec.get("id"), rec.get("subreddit", ""),
                          rec.get("category", ""), n_comments, len(chunks)])
            n += 1
            total_tokens += est
    if reqf:
        reqf.close()

    _save_state(args.workdir, {"model": MODEL, "chunks": chunks})

    est_cost = (total_tokens / 1e6 * 0.05
                + n * OUTPUT_TOKENS_EST / 1e6 * 0.20)
    print(f"Prepared {n:,} requests in {len(chunks)} chunk file(s).")
    for i, c in enumerate(chunks, 1):
        print(f"  chunk {i}: {c['n_requests']:,} requests, "
              f"~{c['est_tokens']/1e6:.1f}M tokens -> {c['file']}")
    print(f"Estimated total batch cost with {MODEL}: ~${est_cost:.2f}")
    print(f"Key index -> {idx_path}")
    print(f"\nNext:  python extract_issues.py submit --workdir {args.workdir} --wait")


# --------------------------------------------------------------------------
# submit / status / jobs / cancel  (verbatim stage-1 machinery)
# --------------------------------------------------------------------------
def _submit_chunk(client, workdir: str, chunk: dict, chunk_no: int) -> None:
    from google.genai import errors as genai_errors

    path = os.path.join(workdir, chunk["file"])
    print(f"Uploading chunk {chunk_no} ({chunk['n_requests']:,} requests, "
          f"~{chunk['est_tokens']/1e6:.1f}M tokens)...")
    uploaded = client.files.upload(
        file=path,
        config={"display_name": f"tdot-issues-{chunk_no:03d}",
                "mime_type": "jsonl"},
    )
    print(f"Creating batch job for chunk {chunk_no}...")
    MAX_TRIES = 20
    RETRY_SECONDS = 60
    job = None
    for attempt in range(1, MAX_TRIES + 1):
        try:
            job = client.batches.create(
                model=MODEL,
                src=uploaded.name,
                config={"display_name": f"tdot-issues-{chunk_no:03d}"},
            )
            break
        except genai_errors.ClientError as e:
            if e.code != 429:
                raise
            if attempt == MAX_TRIES:
                sys.exit(f"Still 429 after {MAX_TRIES} retries -- check the "
                         "Batch panel for a stuck job holding quota, then "
                         "re-run `submit --wait` (it resumes here).")
            print(f"  429 (quota draining) -- retry {attempt}/{MAX_TRIES} "
                  f"in {RETRY_SECONDS}s")
            time.sleep(RETRY_SECONDS)
    chunk["job_name"] = job.name
    chunk["last_state"] = job.state.name
    print(f"  job: {job.name} ({job.state.name})")


def _poll_until_done(client, chunk: dict, interval: int = 60) -> None:
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
    _refresh(client, state)
    _save_state(args.workdir, state)

    for i, chunk in enumerate(state["chunks"], 1):
        if chunk["last_state"] in ACTIVE_STATES:
            if not args.wait:
                sys.exit(f"Chunk {i} is still {chunk['last_state']} -- re-run "
                         f"`submit` when it finishes, or use --wait.")
            print(f"Chunk {i} still running; waiting...")
            _poll_until_done(client, chunk, args.poll_seconds)
            _save_state(args.workdir, state)

        if chunk["job_name"] is None:
            _submit_chunk(client, args.workdir, chunk, i)
            _save_state(args.workdir, state)
            if not args.wait:
                remaining = sum(1 for c in state["chunks"] if c["job_name"] is None)
                print(f"\n{remaining} chunk(s) left." if remaining else
                      "\nLast chunk submitted. Check `status`; then `fetch`.")
                return
            _poll_until_done(client, chunk, args.poll_seconds)
            _save_state(args.workdir, state)
            if chunk["last_state"] != "JOB_STATE_SUCCEEDED":
                sys.exit(f"Chunk {i} ended in {chunk['last_state']} -- fix and "
                         f"re-submit (see `status`).")

    print("\nAll chunks submitted"
          + (" and finished. Run `fetch`." if args.wait else ". Check `status`."))


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
          "\nNot finished. `submit --wait` handles remaining chunks.")


def cmd_jobs(args) -> None:
    client = make_client()
    print("Batch jobs on this account (newest first):")
    shown = 0
    for job in client.batches.list(config={"page_size": min(args.n, 100)}):
        state = job.state.name
        flag = "  <-- HOLDING QUOTA" if state in ACTIVE_STATES else ""
        print(f"  {job.name}  {state:22s} "
              f"{getattr(job, 'display_name', '') or ''}{flag}")
        shown += 1
        if shown >= args.n:
            break


def cmd_cancel(args) -> None:
    client = make_client()
    if args.job:
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
    print(f"{n} active job(s) cancelled." if n else "No active jobs to cancel.")


# --------------------------------------------------------------------------
# fetch: download -> parse JSON -> apply gates -> issues.csv fact table
# --------------------------------------------------------------------------
ISSUE_COLUMNS = [
    "thread_id", "subreddit", "category", "issue_idx",
    "problem_type", "vocab_ok", "program_or_policy",
    "roads", "scope", "place", "anchor_city",
    "statement", "gist", "stance", "stance_ok", "reason",
    "evidence", "evidence_ok",
    "created_utc", "score", "num_comments", "n_issues_in_thread",
]


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
                     f"all chunks must succeed before fetch.")

    # 1. Load the source threads (needed for the evidence gate + metadata)
    threads: dict[str, dict] = {}
    thread_norm: dict[str, str] = {}
    for key, rec in iter_threads(args.threads):
        threads[key] = rec
        text, _ = thread_to_text(rec)
        thread_norm[key] = _norm_text(text)
    print(f"Loaded {len(threads):,} source threads for evidence verification.")

    # 2. Download and parse all results
    raw_out = os.path.join(args.workdir, "issues_raw.jsonl")
    parse_errors: dict[str, str] = {}
    n_issues = 0
    n_threads_done = 0
    n_no_issue = 0
    gate_fail = {"vocab": 0, "evidence": 0, "stance": 0}
    type_counts: dict[str, int] = {}

    with open(args.output, "w", newline="", encoding="utf-8") as fout, \
            open(raw_out, "w", encoding="utf-8") as rawf:
        writer = csv.writer(fout)
        writer.writerow(ISSUE_COLUMNS)
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
                except (KeyError, IndexError, TypeError):
                    parse_errors[key] = "empty/blocked response"
                    continue
                issues, err = parse_issues(text)
                rawf.write(json.dumps({"key": key, "raw": text},
                                      ensure_ascii=False) + "\n")
                if err:
                    parse_errors[key] = err
                    continue
                rec = threads.get(key, {})
                norm = thread_norm.get(key, "")
                n_threads_done += 1
                if not issues:
                    n_no_issue += 1
                    continue
                for idx_i, iss in enumerate(issues, 1):
                    v = validate_issue(iss, norm)
                    if not v["vocab_ok"]:
                        gate_fail["vocab"] += 1
                    if not v["evidence_ok"]:
                        gate_fail["evidence"] += 1
                    if not v["stance_ok"]:
                        gate_fail["stance"] += 1
                    type_counts[v["problem_type"]] = \
                        type_counts.get(v["problem_type"], 0) + 1
                    writer.writerow([
                        rec.get("id", key), rec.get("subreddit", ""),
                        rec.get("category", ""), idx_i,
                        v["problem_type"], v["vocab_ok"], v["program_or_policy"],
                        v["roads"], v["scope"], v["place"], v["anchor_city"],
                        v["statement"], v["gist"], v["stance"], v["stance_ok"],
                        v["reason"], v["evidence"], v["evidence_ok"],
                        rec.get("created_utc", ""), rec.get("score", ""),
                        rec.get("num_comments", ""), len(issues),
                    ])
                    n_issues += 1

    # 3. Error ids file for a rescue pass
    err_path = os.path.join(args.workdir, "error_ids.txt")
    with open(err_path, "w", encoding="utf-8") as f:
        for key, reason in parse_errors.items():
            rid = key.split("_", 1)[1] if "_" in key else key
            f.write(f"{rid}\t{reason}\n")

    # 4. Report
    print(f"\nWrote {n_issues:,} issue records from {n_threads_done:,} threads "
          f"-> {args.output}")
    print(f"Raw model JSON kept in {raw_out}")
    print(f"Threads with no issue reported: {n_no_issue:,}")
    print(f"Parse failures: {len(parse_errors):,} (ids -> {err_path}; rescue: "
          f"prepare --ids-file {err_path} --workdir <new_dir>)")
    print(f"Gate failures: {gate_fail['vocab']} unknown problem_type, "
          f"{gate_fail['evidence']} evidence-not-found (hallucination flags), "
          f"{gate_fail['stance']} bad stance values")
    n_other = type_counts.get("OTHER", 0)
    if n_issues:
        print(f"OTHER rate: {n_other}/{n_issues} = {n_other/max(n_issues,1):.1%} "
              f"(vocabulary coverage diagnostic -- inspect OTHER gists if >5%)")
    print("\nTop problem types:")
    for t, c in sorted(type_counts.items(), key=lambda kv: -kv[1])[:15]:
        print(f"  {c:5,}  {t}")
    print("\nAudit habit: sample rows where evidence_ok=False or "
          "problem_type=OTHER and read them with view_thread.py.")


# --------------------------------------------------------------------------
# errors (batch-level diagnosis, same as stage 1)
# --------------------------------------------------------------------------
def cmd_errors(args) -> None:
    client = make_client()
    state = _load_state(args.workdir)
    reasons: dict[str, int] = {}
    bad: list[tuple[str, str]] = []
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
                _, err = parse_issues(text)
                if err:
                    reason = err
            except (KeyError, IndexError, TypeError):
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


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2: issue extraction on classified TDOT threads "
                    "(Gemini Batch API)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("test", help="realtime sanity check with live gate checks")
    p.add_argument("--threads", required=True,
                   help="threads_related.jsonl (with label/category)")
    p.add_argument("-n", type=int, default=5)
    p.add_argument("--id", nargs="+", default=None)
    p.add_argument("--model", default=None,
                   help=f"override model for this test only (default {MODEL})")

    p = sub.add_parser("prepare", help="build chunked batch request files")
    p.add_argument("--threads", required=True)
    p.add_argument("--workdir", required=True)
    p.add_argument("--chunk-tokens", type=int, default=CHUNK_TOKEN_LIMIT)
    p.add_argument("--ids-file", default=None,
                   help="only these thread ids (e.g. error_ids.txt rescue)")

    p = sub.add_parser("submit", help="upload + create batch job(s)")
    p.add_argument("--workdir", required=True)
    p.add_argument("--wait", action="store_true")
    p.add_argument("--poll-seconds", type=int, default=60)

    p = sub.add_parser("status", help="check job state")
    p.add_argument("--workdir", required=True)

    p = sub.add_parser("jobs", help="list batch jobs on the account")
    p.add_argument("-n", type=int, default=20)

    p = sub.add_parser("cancel", help="terminate running jobs")
    p.add_argument("--workdir", default=None)
    p.add_argument("--job", default=None)

    p = sub.add_parser("errors", help="explain failed requests")
    p.add_argument("--workdir", required=True)

    p = sub.add_parser("fetch", help="download -> validate -> issues.csv")
    p.add_argument("--workdir", required=True)
    p.add_argument("--threads", required=True,
                   help="threads_related.jsonl again (for the evidence gate)")
    p.add_argument("--output", required=True, help="issues.csv path")
    p.add_argument("--wait", action="store_true")
    p.add_argument("--poll-seconds", type=int, default=60)

    args = parser.parse_args()
    if args.command == "cancel" and not args.workdir and not args.job:
        sys.exit("cancel: provide --workdir or --job batches/xxxx")
    {"test": cmd_test, "prepare": cmd_prepare, "submit": cmd_submit,
     "status": cmd_status, "jobs": cmd_jobs, "cancel": cmd_cancel,
     "errors": cmd_errors, "fetch": cmd_fetch}[args.command](args)


if __name__ == "__main__":
    main()
