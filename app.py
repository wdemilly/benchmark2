import io
import json
import time
import zipfile
import hashlib
import shutil
import re
import os
import base64
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Iterable

import pandas as pd
import requests
import streamlit as st

try:
    import anthropic  # type: ignore
except Exception:
    anthropic = None

APP_TITLE = "Micro-Prompt Harness"
DATA_DIR = Path("micro_prompt_runs")
DATA_DIR.mkdir(exist_ok=True)

OUTPUTS_DIR = DATA_DIR / "flat_outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

DEFAULT_BASE_PROMPT = '''Read all attached documents from beginning to end. Do not sample them. Then follow the instructions in the micro-prompt below to write the chapter. Write the chapter straight through in one continuous pass. Return plain text only, with normal paragraph breaks and no commentary.'''
DEFAULT_MODEL = "claude-opus-4-6"
DEFAULT_EVALUATOR_MODEL = "claude-opus-4-6"
DEFAULT_MAX_TOKENS = 12000
MAX_ALLOWED_TOKENS = 32000
MAX_CONTINUATIONS = 4
PROMPTS_CSV = Path("prompts.csv")
CURRENT_SESSION_KEY = "app_session_id"
CURRENT_SESSION_RUN_IDS_KEY = "app_session_run_ids"
LATEST_BATCH_RUN_IDS_KEY = "app_latest_batch_run_ids"

EVALUATOR_PROMPT = '''You are reading N drafts of the same chapter of a novel. They were generated from the same source material and outline. Your job has two parts.

PART ONE: pick the best draft, as a reader of this genre would.

Read every draft in full. Do not skim. Before judging, infer the project's register from the drafts themselves: genre and subgenre, period, point of view, tense, narrator's class and position, and established voice. Hold the drafts to their own standard, not to a generic "literary" ideal.

Then judge as an experienced reader of this specific genre would judge -- the kind of reader who buys new novels in this space, knows the conventions, and can tell a fresh sentence from a familiar one. Your question is simple: which of these drafts would that reader most want to keep reading?

In answering, give weight to:

- Specificity over atmosphere. Reward drafts where the writer commits to a concrete observed detail (an object, a gesture, a number, a name) instead of reaching for mood or generalization. Penalize drafts that produce the sensation of good writing through cadence alone.
- Render over interpret. The strongest draft shows; weaker drafts explain. Penalize drafts that name emotions, summarize their own meaning, or close paragraphs or scenes with an interpretive sentence telling the reader what just happened.
- Dialogue doing dramatic work. Lines must carry tension, subtext, character, or forward motion -- ideally more than one at once. Penalize polite turns stating positions, or exposition in quotation marks.
- Voice consistency. The established voice must hold throughout without drift, pastiche, or anachronism.
- Trust in the ending. The strongest draft ends on the beat the scene has earned and stops. Weaker drafts add a coda explaining or softening the beat.
- Restraint with simile, aphorism, and summary. Reward sentences that leave the reader to do the work.

Be a demanding reader. Do not be diplomatic. If two drafts are close, say what specifically tips the decision.

PART TWO: scavenge the losing drafts for transplantable material.

After you have chosen the winner, go back through the non-winning drafts and find specific lines, phrases, images, or small passages that are better than what the winner has in the equivalent spot, and which could be grafted into the winner with minimal rewriting. An "average" draft may contain the single best line in the batch; that is what you are hunting for.

For each transplant candidate:
- Quote the source text exactly (a line or a short passage, not a whole paragraph).
- Name the source draft number.
- Describe in one sentence where in the winner this should go -- which paragraph or scene beat it should replace or be inserted into.
- Briefly say why it is stronger than what the winner has there.

Only propose a transplant if you genuinely believe the winner would be improved by it. Do not pad the list. Three or four strong transplant candidates is better than a dozen weak ones. Zero is a legitimate answer if the winner really is cleanly superior at every beat.

OUTPUT FORMAT

Produce your evaluation in this exact structure:

1. A short paragraph (4-7 sentences) per draft, assessing its strengths and weaknesses as a reader of the genre would. Quote specific sentences, for both failures and successes.

2. A comparison paragraph naming the two or three closest contenders and the specific reasons one edges out the others.

3. A section with the heading:

TRANSPLANTS

Under this heading, list each transplant candidate as a numbered item. Use this template exactly for each item:

1. FROM DRAFT N: "<quoted text>"
   GRAFT INTO: <where in the winner this belongs -- paragraph or beat>
   WHY: <one sentence>

If there are no transplants worth proposing, write: (none)

4. On a line by itself:

RANKING: N, N, N, ...

where the numbers are every draft number in order from strongest to weakest, separated by commas. Include every draft exactly once.

5. On the final line of your response:

WINNER: N

where N is the number of the winning draft. Nothing after that line. The words TRANSPLANTS, RANKING, and WINNER must appear in all caps.
'''


def ensure_session_state() -> str:
    if CURRENT_SESSION_KEY not in st.session_state:
        st.session_state[CURRENT_SESSION_KEY] = datetime.now().strftime("%Y%m%d_%H%M%S")
    if CURRENT_SESSION_RUN_IDS_KEY not in st.session_state:
        st.session_state[CURRENT_SESSION_RUN_IDS_KEY] = []
    if LATEST_BATCH_RUN_IDS_KEY not in st.session_state:
        st.session_state[LATEST_BATCH_RUN_IDS_KEY] = []
    return str(st.session_state[CURRENT_SESSION_KEY])


@dataclass
class RunRecord:
    run_id: str
    session_id: str
    timestamp: str
    batch_label: str
    prompt_id: int
    repetition_index: int
    category: str
    provider: str
    model: str
    temperature: float
    max_tokens: int
    continuation_rounds: int
    source_name: str
    outline_name: str
    profiles_name: str
    file_stub: str
    output_file: str
    payload_file: str
    micro_prompt_file: str
    meta_file: str
    output_sha256: str = ""
    stop_reason: str = ""
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    output_words: Optional[int] = None
    truncation_flag: bool = False
    evaluation_id: str = ""
    is_winner: bool = False
    evaluation_rank: Optional[int] = None
    evaluation_raw: str = ""
    evaluation_parse_status: str = ""
    evaluator_model: str = ""
    originality_label: str = ""
    originality_score: Optional[float] = None
    manual_rating: str = ""
    manual_notes: str = ""
    signature_score: Optional[int] = None
    signature_report: str = ""


def load_prompt_definitions(csv_path: Path) -> List[dict]:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {csv_path}. Place prompts.csv beside app.py."
        )

    df = pd.read_csv(csv_path)

    required_columns = ["id", "category", "text"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"Prompt file is missing required column(s): {', '.join(missing)}"
        )

    if df.empty:
        raise ValueError("Prompt file is empty.")

    prompts: List[dict] = []
    seen_ids = set()

    for row_number, row in df.iterrows():
        raw_id = row["id"]
        raw_category = row["category"]
        raw_text = row["text"]

        if pd.isna(raw_id):
            raise ValueError(f"Row {row_number + 2}: id is blank.")
        if pd.isna(raw_category) or not str(raw_category).strip():
            raise ValueError(f"Row {row_number + 2}: category is blank.")
        if pd.isna(raw_text) or not str(raw_text).strip():
            raise ValueError(f"Row {row_number + 2}: text is blank.")

        try:
            prompt_id = int(raw_id)
        except Exception as exc:
            raise ValueError(f"Row {row_number + 2}: id must be an integer.") from exc

        if prompt_id in seen_ids:
            raise ValueError(f"Duplicate prompt id found: {prompt_id}")
        seen_ids.add(prompt_id)

        prompts.append(
            {
                "id": prompt_id,
                "category": str(raw_category).strip(),
                "text": str(raw_text).strip(),
            }
        )

    prompts.sort(key=lambda p: p["id"])
    return prompts


def normalize_text(text: str) -> str:
    return (
        text.replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\u2013", "-")
        .replace("\u2014", "--")
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u00a0", " ")
    )


def normalize_output_text(text: str) -> str:
    text = normalize_text(text)
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip() + "\n"


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def decode_uploaded_text(uploaded_file) -> str:
    raw = uploaded_file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    return normalize_text(text)


def extract_text_from_response(resp) -> str:
    parts: List[str] = []
    for block in getattr(resp, "content", []) or []:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "".join(parts)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------
# GitHub sync
#
# Best-effort overlay on top of the local filesystem. On app startup the
# contents of the configured repo are pulled down into DATA_DIR. After every
# successful generation or evaluation the changed files are pushed back up.
# Uses the GitHub REST contents API directly (one call per file) to avoid a
# PyGithub dependency. Failures are logged but never block generation.
# ----------------------------------------------------------------------------

GITHUB_API_BASE = "https://api.github.com"
GITHUB_SYNC_STATUS_KEY = "github_sync_status"
GITHUB_PULLED_KEY = "github_pulled_this_session"


def load_github_config() -> dict:
    """Read GitHub sync config from Streamlit secrets first, then env vars.

    Returns a dict with keys: token, repo, branch, configured (bool),
    source (str describing where config was found).
    """
    token = ""
    repo = ""
    branch = ""
    source = ""

    try:
        if "GITHUB_TOKEN" in st.secrets:
            token = str(st.secrets.get("GITHUB_TOKEN", "")).strip()
            repo = str(st.secrets.get("GITHUB_REPO", "")).strip()
            branch = str(st.secrets.get("GITHUB_BRANCH", "") or "main").strip()
            if token and repo:
                source = "Streamlit secrets"
    except Exception:
        token = ""
        repo = ""

    if not (token and repo):
        env_token = os.environ.get("GITHUB_TOKEN", "").strip()
        env_repo = os.environ.get("GITHUB_REPO", "").strip()
        env_branch = os.environ.get("GITHUB_BRANCH", "main").strip() or "main"
        if env_token and env_repo:
            token = env_token
            repo = env_repo
            branch = env_branch
            source = "environment variable"

    configured = bool(token and repo)
    return {
        "token": token,
        "repo": repo,
        "branch": branch or "main",
        "configured": configured,
        "source": source,
    }


def _gh_headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _gh_record_status(message: str, kind: str = "info") -> None:
    st.session_state[GITHUB_SYNC_STATUS_KEY] = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "message": message,
        "kind": kind,
    }


def github_list_tree(cfg: dict) -> List[dict]:
    """List every file in the repo at cfg['branch'].

    Returns a list of dicts with keys 'path' and 'sha'. Returns an empty
    list if the branch is empty or the call fails.
    """
    if not cfg.get("configured"):
        return []
    repo = cfg["repo"]
    branch = cfg["branch"]

    # Resolve branch -> tree sha via the branches endpoint.
    try:
        branch_resp = requests.get(
            f"{GITHUB_API_BASE}/repos/{repo}/branches/{branch}",
            headers=_gh_headers(cfg["token"]),
            timeout=15,
        )
    except requests.RequestException as exc:
        _gh_record_status(f"GitHub list failed: {exc}", kind="error")
        return []

    if branch_resp.status_code == 404:
        # Branch does not exist yet (fresh repo). Nothing to pull.
        return []
    if not branch_resp.ok:
        _gh_record_status(
            f"GitHub list failed: {branch_resp.status_code} {branch_resp.text[:200]}",
            kind="error",
        )
        return []

    tree_sha = (
        branch_resp.json().get("commit", {}).get("commit", {}).get("tree", {}).get("sha")
    )
    if not tree_sha:
        return []

    try:
        tree_resp = requests.get(
            f"{GITHUB_API_BASE}/repos/{repo}/git/trees/{tree_sha}",
            params={"recursive": "1"},
            headers=_gh_headers(cfg["token"]),
            timeout=30,
        )
    except requests.RequestException as exc:
        _gh_record_status(f"GitHub tree read failed: {exc}", kind="error")
        return []

    if not tree_resp.ok:
        _gh_record_status(
            f"GitHub tree read failed: {tree_resp.status_code} {tree_resp.text[:200]}",
            kind="error",
        )
        return []

    entries = tree_resp.json().get("tree", []) or []
    return [
        {"path": entry["path"], "sha": entry["sha"]}
        for entry in entries
        if entry.get("type") == "blob" and entry.get("path")
    ]


def github_get_file_bytes(cfg: dict, path: str) -> Optional[bytes]:
    """Fetch a single file's bytes from the repo. Returns None on failure."""
    if not cfg.get("configured"):
        return None
    try:
        resp = requests.get(
            f"{GITHUB_API_BASE}/repos/{cfg['repo']}/contents/{path}",
            params={"ref": cfg["branch"]},
            headers=_gh_headers(cfg["token"]),
            timeout=30,
        )
    except requests.RequestException:
        return None
    if not resp.ok:
        return None
    body = resp.json()
    # For files under ~1MB the response includes base64 content directly.
    if body.get("encoding") == "base64" and "content" in body:
        try:
            return base64.b64decode(body["content"])
        except Exception:
            return None
    # For larger files, fall back to the download_url.
    download_url = body.get("download_url")
    if download_url:
        try:
            dl = requests.get(download_url, timeout=60)
            if dl.ok:
                return dl.content
        except requests.RequestException:
            return None
    return None


def github_get_file_sha(cfg: dict, path: str) -> Optional[str]:
    """Return the blob sha of a file at the tip of the branch, or None if it
    does not exist. Needed for updates (the API requires the current sha to
    prevent races)."""
    if not cfg.get("configured"):
        return None
    try:
        resp = requests.get(
            f"{GITHUB_API_BASE}/repos/{cfg['repo']}/contents/{path}",
            params={"ref": cfg["branch"]},
            headers=_gh_headers(cfg["token"]),
            timeout=15,
        )
    except requests.RequestException:
        return None
    if resp.status_code == 404:
        return None
    if not resp.ok:
        return None
    return resp.json().get("sha")


def github_put_file(cfg: dict, path: str, data: bytes, message: str) -> bool:
    """Create or update a single file in the repo. Returns True on success."""
    if not cfg.get("configured"):
        return False

    existing_sha = github_get_file_sha(cfg, path)

    payload = {
        "message": message,
        "content": base64.b64encode(data).decode("ascii"),
        "branch": cfg["branch"],
    }
    if existing_sha:
        payload["sha"] = existing_sha

    try:
        resp = requests.put(
            f"{GITHUB_API_BASE}/repos/{cfg['repo']}/contents/{path}",
            headers=_gh_headers(cfg["token"]),
            json=payload,
            timeout=30,
        )
    except requests.RequestException as exc:
        _gh_record_status(f"GitHub push failed for {path}: {exc}", kind="error")
        return False

    if not resp.ok:
        _gh_record_status(
            f"GitHub push failed for {path}: {resp.status_code} {resp.text[:200]}",
            kind="error",
        )
        return False
    return True


def _local_path_for_repo_path(repo_path: str) -> Path:
    """Map a repo-relative path back to its local filesystem path under DATA_DIR."""
    return DATA_DIR / repo_path


def _repo_path_for_local(local_path: Path) -> Optional[str]:
    """Map a local path under DATA_DIR back to a forward-slash repo path.
    Returns None if the file is not under DATA_DIR."""
    try:
        rel = local_path.resolve().relative_to(DATA_DIR.resolve())
    except Exception:
        return None
    return rel.as_posix()


def github_pull_all(cfg: dict) -> dict:
    """Pull every file from the repo into DATA_DIR, overwriting local copies.

    Returns a dict with counts: pulled, skipped, failed.
    """
    result = {"pulled": 0, "skipped": 0, "failed": 0}
    if not cfg.get("configured"):
        return result

    tree = github_list_tree(cfg)
    if not tree:
        _gh_record_status("Pull: no files in repo (or repo is empty).", kind="info")
        return result

    for entry in tree:
        repo_path = entry["path"]
        local_path = _local_path_for_repo_path(repo_path)
        data = github_get_file_bytes(cfg, repo_path)
        if data is None:
            result["failed"] += 1
            continue
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(data)
            result["pulled"] += 1
        except Exception:
            result["failed"] += 1

    _gh_record_status(
        f"Pulled {result['pulled']} file(s) from {cfg['repo']}@{cfg['branch']}.",
        kind="success" if result["failed"] == 0 else "warn",
    )
    return result


def github_push_paths(cfg: dict, local_paths: List[Path], commit_prefix: str) -> dict:
    """Push a list of local files to the repo. Returns dict with pushed/failed counts."""
    result = {"pushed": 0, "failed": 0}
    if not cfg.get("configured"):
        return result

    for local_path in local_paths:
        if not local_path.exists() or not local_path.is_file():
            continue
        repo_path = _repo_path_for_local(local_path)
        if repo_path is None:
            continue
        try:
            data = local_path.read_bytes()
        except Exception:
            result["failed"] += 1
            continue
        commit_msg = f"{commit_prefix}: {repo_path}"
        ok = github_put_file(cfg, repo_path, data, commit_msg)
        if ok:
            result["pushed"] += 1
        else:
            result["failed"] += 1

    if result["pushed"] or result["failed"]:
        kind = "success" if result["failed"] == 0 else "warn"
        _gh_record_status(
            f"Pushed {result['pushed']} file(s) to {cfg['repo']}"
            + (f" ({result['failed']} failed)" if result["failed"] else ""),
            kind=kind,
        )
    return result


def github_push_after_generation(
    cfg: dict,
    csv_path: Path,
    output_path: Path,
    payload_path: Path,
    micro_prompt_path: Path,
    meta_path: Path,
) -> None:
    """Push the files produced by a single generation run plus the updated CSV."""
    if not cfg.get("configured"):
        return
    github_push_paths(
        cfg,
        [csv_path, output_path, payload_path, micro_prompt_path, meta_path],
        commit_prefix="generation",
    )


def github_push_after_evaluation(
    cfg: dict,
    csv_path: Path,
    winner_path: Path,
    transplants_path: Optional[Path],
) -> None:
    """Push the files produced by an evaluation plus the updated CSV."""
    if not cfg.get("configured"):
        return
    paths = [csv_path, winner_path]
    if transplants_path and transplants_path.exists():
        paths.append(transplants_path)
    github_push_paths(cfg, paths, commit_prefix="evaluation")


def github_pull_on_startup_if_needed(cfg: dict, csv_path: Path) -> None:
    """If the repo is configured and we have not yet pulled in this session, pull."""
    if not cfg.get("configured"):
        return
    if st.session_state.get(GITHUB_PULLED_KEY):
        return
    # Only auto-pull if the local CSV is missing or empty. This avoids
    # clobbering in-progress local work if the app is restarted mid-session.
    local_empty = (not csv_path.exists()) or csv_path.stat().st_size == 0
    if local_empty:
        github_pull_all(cfg)
    st.session_state[GITHUB_PULLED_KEY] = True


def load_records(csv_path: Path) -> pd.DataFrame:
    columns = [field for field in RunRecord.__dataclass_fields__.keys()]

    text_columns = [
        "run_id",
        "session_id",
        "timestamp",
        "batch_label",
        "category",
        "provider",
        "model",
        "source_name",
        "outline_name",
        "profiles_name",
        "file_stub",
        "output_file",
        "payload_file",
        "micro_prompt_file",
        "meta_file",
        "output_sha256",
        "stop_reason",
        "evaluation_id",
        "evaluation_raw",
        "evaluation_parse_status",
        "evaluator_model",
        "originality_label",
        "manual_rating",
        "manual_notes",
        "signature_report",
    ]

    bool_columns = [
        "truncation_flag",
        "is_winner",
    ]

    numeric_columns = [
        "prompt_id",
        "repetition_index",
        "temperature",
        "max_tokens",
        "continuation_rounds",
        "input_tokens",
        "output_tokens",
        "output_words",
        "originality_score",
        "evaluation_rank",
        "signature_score",
    ]

    if csv_path.exists():
        df = pd.read_csv(csv_path)

        for col in columns:
            if col not in df.columns:
                df[col] = pd.NA

        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype("object")

        for col in bool_columns:
            if col in df.columns:
                df[col] = df[col].astype("object")

        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df[columns]

    return pd.DataFrame(columns=columns)


def append_record(csv_path: Path, record: RunRecord) -> None:
    df = load_records(csv_path)
    new_row = pd.DataFrame([asdict(record)])

    for col in df.columns:
        if col in new_row.columns and df[col].dtype == object:
            new_row[col] = new_row[col].astype("object")

    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(csv_path, index=False)


def update_record(csv_path: Path, run_id: str, updates: dict) -> None:
    df = load_records(csv_path)
    if df.empty:
        return
    mask = df["run_id"].astype(str) == str(run_id)
    if not mask.any():
        return
    for key, value in updates.items():
        if key in df.columns:
            if isinstance(value, str):
                df[key] = df[key].astype("object")
            df.loc[mask, key] = value
    df.to_csv(csv_path, index=False)


def clear_all_run_data(csv_path: Path, outputs_root: Path) -> None:
    if csv_path.exists():
        csv_path.unlink()
    if outputs_root.exists():
        for item in outputs_root.iterdir():
            if item.is_file() or item.is_symlink():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
    outputs_root.mkdir(parents=True, exist_ok=True)
    st.session_state[CURRENT_SESSION_RUN_IDS_KEY] = []
    st.session_state[CURRENT_SESSION_KEY] = datetime.now().strftime("%Y%m%d_%H%M%S")


def build_payload(base_prompt: str, micro_prompt: str, source_text: str, outline_text: str, profiles_text: str) -> str:
    parts = [
        "BASE PROMPT",
        base_prompt.strip(),
        "",
        "TEST MICRO-PROMPT",
        micro_prompt.strip(),
        "",
        "BEGIN COMBINED SOURCE TEXTS",
        source_text.strip(),
        "END COMBINED SOURCE TEXTS",
        "",
        "BEGIN OUTLINE",
        outline_text.strip(),
        "END OUTLINE",
    ]
    if profiles_text.strip():
        parts.extend([
            "",
            "BEGIN CHARACTER PROFILES",
            profiles_text.strip(),
            "END CHARACTER PROFILES",
        ])
    parts.extend([
        "",
        "Write the full chapter now. Return plain text only, with normal paragraph breaks and no commentary."
    ])
    return "\n".join(parts)


def get_usage_tokens(resp) -> Tuple[Optional[int], Optional[int]]:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return None, None
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    return input_tokens, output_tokens


def anthropic_messages_create_with_backoff(
    client,
    *,
    model: str,
    max_tokens: int,
    temperature: float,
    messages: list,
    max_attempts: int = 6,
    initial_delay: float = 2.0,
):
    last_exc = None
    delay = initial_delay

    for attempt in range(1, max_attempts + 1):
        try:
            return client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
            )
        except Exception as exc:
            last_exc = exc
            message = str(exc).lower()

            retryable = any(
                token in message
                for token in [
                    "rate limit",
                    "rate_limit",
                    "overloaded",
                    "timeout",
                    "timed out",
                    "connection",
                    "529",
                    "502",
                    "503",
                    "504",
                ]
            )

            if not retryable or attempt >= max_attempts:
                raise

            time.sleep(delay)
            delay = min(delay * 2, 20.0)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Unknown Anthropic API failure.")


def call_anthropic_with_continuation(
    api_key: str,
    model: str,
    payload: str,
    max_tokens: int,
    temperature: float,
    max_continuations: int = MAX_CONTINUATIONS,
) -> dict:
    if anthropic is None:
        raise RuntimeError("anthropic package is not installed.")

    client = anthropic.Anthropic(api_key=api_key)

    messages = [{"role": "user", "content": payload}]
    collected_parts: List[str] = []
    total_input_tokens = 0
    total_output_tokens = 0
    continuation_rounds = 0
    final_stop_reason = ""
    last_response_id = ""

    for round_index in range(max_continuations + 1):
        resp = anthropic_messages_create_with_backoff(
            client,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
        )

        text_chunk = extract_text_from_response(resp)
        if not text_chunk.strip():
            raise RuntimeError("Model returned an empty text block.")

        text_chunk = normalize_output_text(text_chunk)
        collected_parts.append(text_chunk)

        input_tokens, output_tokens = get_usage_tokens(resp)
        if input_tokens is not None:
            total_input_tokens += int(input_tokens)
        if output_tokens is not None:
            total_output_tokens += int(output_tokens)

        final_stop_reason = str(getattr(resp, "stop_reason", "") or "")
        last_response_id = str(getattr(resp, "id", "") or "")

        if final_stop_reason == "max_tokens":
            continuation_rounds += 1
            if round_index >= max_continuations:
                break

            messages.append({"role": "assistant", "content": text_chunk})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Continue exactly where you left off. Do not restart. "
                        "Do not summarize. Do not add commentary. "
                        "Continue the chapter in plain text only."
                    ),
                }
            )
            time.sleep(1.0)
            continue

        break

    combined_text = "".join(collected_parts)
    combined_text = normalize_output_text(combined_text)
    truncation_flag = final_stop_reason == "max_tokens"
    output_words = len(combined_text.split())

    return {
        "text": combined_text,
        "stop_reason": final_stop_reason,
        "input_tokens": total_input_tokens or None,
        "output_tokens": total_output_tokens or None,
        "output_words": output_words,
        "truncation_flag": truncation_flag,
        "continuation_rounds": continuation_rounds,
        "response_id": last_response_id,
    }


def parse_winner_integer(text: str, max_valid: int) -> Tuple[Optional[int], str]:
    cleaned = (text or "").strip()
    if not cleaned:
        return None, "failed"

    # Preferred: explicit 'WINNER: N' declaration (case-insensitive; last one wins)
    winner_matches = list(re.finditer(r"WINNER\s*[:\-]\s*(\d+)", cleaned, re.IGNORECASE))
    if winner_matches:
        value = int(winner_matches[-1].group(1))
        if 1 <= value <= max_valid:
            return value, "clean"

    if re.fullmatch(r"\d+[.\s]*", cleaned):
        value = int(re.search(r"\d+", cleaned).group(0))
        if 1 <= value <= max_valid:
            return value, "clean"
        return None, "failed"

    # Last-resort: the final integer in the response, if in valid range
    all_ints = list(re.finditer(r"\b(\d+)\b", cleaned))
    for match in reversed(all_ints):
        value = int(match.group(1))
        if 1 <= value <= max_valid:
            return value, "parsed"

    return None, "failed"


def parse_ranking_list(text: str, max_valid: int) -> List[int]:
    """Parse a 'RANKING: N, N, N' line into an ordered list of draft indices.
    Returns the best-to-worst ranking with duplicates removed in order. Any
    missing drafts are appended at the end in ascending order so the list is
    always a permutation of 1..max_valid. Returns [] if no ranking line is found."""
    cleaned = (text or "").strip()
    if not cleaned:
        return []

    match = re.search(r"RANKING\s*[:\-]\s*([0-9,\s]+)", cleaned, re.IGNORECASE)
    if not match:
        return []

    raw_list = match.group(1)
    found: List[int] = []
    seen = set()
    for num_str in re.findall(r"\d+", raw_list):
        value = int(num_str)
        if 1 <= value <= max_valid and value not in seen:
            found.append(value)
            seen.add(value)

    if not found:
        return []

    for i in range(1, max_valid + 1):
        if i not in seen:
            found.append(i)

    return found


def parse_transplants(text: str, max_draft: int) -> List[dict]:
    """Parse the TRANSPLANTS section of the evaluator's response.

    Looks for the 'TRANSPLANTS' heading and reads numbered items in the
    prescribed format:

        N. FROM DRAFT N: "<quoted text>"
           GRAFT INTO: <location>
           WHY: <one sentence>

    Returns a list of dicts with keys: source_draft, quote, graft_into, why.
    Returns [] if no section is found, if it says '(none)', or if no items
    parse cleanly.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return []

    # Isolate the section between TRANSPLANTS and the next all-caps section
    # heading (RANKING, WINNER, or end-of-string).
    section_match = re.search(
        r"TRANSPLANTS\s*\n(.*?)(?=\n\s*(?:RANKING|WINNER)\s*[:\-]|\Z)",
        cleaned,
        re.IGNORECASE | re.DOTALL,
    )
    if not section_match:
        return []

    section = section_match.group(1).strip()
    if not section or re.fullmatch(r"\(?\s*none\s*\)?\.?", section, re.IGNORECASE):
        return []

    # Each item starts with a number followed by ". FROM DRAFT".
    # Use a lookahead to split on item boundaries.
    item_pattern = re.compile(
        r"(?:^|\n)\s*\d+\.\s+FROM\s+DRAFT\s+(\d+)\s*[:\-]\s*(.*?)(?=\n\s*\d+\.\s+FROM\s+DRAFT\s+\d+|\Z)",
        re.IGNORECASE | re.DOTALL,
    )

    transplants: List[dict] = []
    for match in item_pattern.finditer(section):
        draft_num = int(match.group(1))
        if draft_num < 1 or draft_num > max_draft:
            continue
        body = match.group(2).strip()

        # Extract the quoted text. Accept straight or curly quotes.
        quote_match = re.search(r'["\u201c]([^"\u201d]+?)["\u201d]', body)
        quote = quote_match.group(1).strip() if quote_match else ""

        # Pull GRAFT INTO and WHY fields.
        graft_match = re.search(
            r"GRAFT\s+INTO\s*[:\-]\s*(.+?)(?=\n\s*WHY\s*[:\-]|\Z)",
            body,
            re.IGNORECASE | re.DOTALL,
        )
        graft_into = graft_match.group(1).strip() if graft_match else ""

        why_match = re.search(
            r"WHY\s*[:\-]\s*(.+?)\Z",
            body,
            re.IGNORECASE | re.DOTALL,
        )
        why = why_match.group(1).strip() if why_match else ""

        # Only keep items that have a quote and at least one of graft/why.
        if quote and (graft_into or why):
            transplants.append(
                {
                    "source_draft": draft_num,
                    "quote": quote,
                    "graft_into": re.sub(r"\s+", " ", graft_into),
                    "why": re.sub(r"\s+", " ", why),
                }
            )

    return transplants


def evaluate_drafts_with_anthropic(
    api_key: str,
    model: str,
    drafts: List[Tuple[str, str]],
) -> dict:
    if anthropic is None:
        raise RuntimeError("anthropic package is not installed.")

    if not model or not model.strip():
        raise RuntimeError("Evaluator model is blank.")

    if len(drafts) < 2:
        raise RuntimeError("Evaluation requires at least 2 drafts.")

    client = anthropic.Anthropic(api_key=api_key)

    draft_blocks = []
    for index, (_run_id, text) in enumerate(drafts, start=1):
        draft_blocks.append(
            f"DRAFT {index}\n\n"
            f"{text.strip()}"
        )

    payload = (
        f"{EVALUATOR_PROMPT}\n\n"
        + "\n\n---\n\n".join(draft_blocks)
    )

    # Pre-flight size check. Claude Opus 4.6 has a 200k-token context window.
    # Reserve 8k tokens for the response and per-message overhead.
    # Rough char-to-token ratio for English prose is ~4 chars per token; we use 3.5
    # to be conservative (overestimate tokens so we warn earlier rather than later).
    CONTEXT_WINDOW_TOKENS = 200_000
    RESPONSE_RESERVE_TOKENS = 8_000
    INPUT_BUDGET_TOKENS = CONTEXT_WINDOW_TOKENS - RESPONSE_RESERVE_TOKENS

    estimated_input_tokens = int(len(payload) / 3.5)
    if estimated_input_tokens > INPUT_BUDGET_TOKENS:
        total_words = sum(len(t.split()) for _, t in drafts)
        avg_words = total_words // max(len(drafts), 1)
        raise RuntimeError(
            f"Evaluator input is too large for the model's context window. "
            f"Estimated {estimated_input_tokens:,} input tokens "
            f"(budget is {INPUT_BUDGET_TOKENS:,} after reserving "
            f"{RESPONSE_RESERVE_TOKENS:,} for the response). "
            f"Batch has {len(drafts)} drafts totaling {total_words:,} words "
            f"(avg {avg_words:,} words/draft). "
            f"Evaluate fewer drafts at once or shorten the inputs."
        )

    resp = anthropic_messages_create_with_backoff(
        client,
        model=model.strip(),
        max_tokens=6000,
        temperature=0,
        messages=[{"role": "user", "content": payload}],
        max_attempts=5,
        initial_delay=2.0,
    )

    raw_text = extract_text_from_response(resp)
    winner_index, parse_status = parse_winner_integer(raw_text, max_valid=len(drafts))

    if winner_index is None:
        raise RuntimeError(f"Could not parse a valid draft number from evaluator response: {raw_text!r}")

    winner_run_id = drafts[winner_index - 1][0]
    ranking_list = parse_ranking_list(raw_text, max_valid=len(drafts))
    transplants = parse_transplants(raw_text, max_draft=len(drafts))

    # Tag each transplant with the source run_id for downstream display/saving.
    for tp in transplants:
        src_idx = tp["source_draft"] - 1
        if 0 <= src_idx < len(drafts):
            tp["source_run_id"] = drafts[src_idx][0]
        else:
            tp["source_run_id"] = ""

    return {
        "winner_run_id": winner_run_id,
        "winner_index": winner_index,
        "ranking": ranking_list,
        "raw_text": raw_text.strip(),
        "parse_status": parse_status,
        "model": model.strip(),
        "transplants": transplants,
    }


GRAFT_INSTRUCTIONS = """You are a careful line editor performing transplant grafts on a prose chapter.

You will receive:
  1. The WINNING CHAPTER (the base text).
  2. A list of TRANSPLANT CANDIDATES, each with a quoted line from a losing draft, a target location in the winner, and a rationale.

Your job is to produce a single revised chapter that integrates the transplants into the winner.

RULES:
- Begin with the winner's text as the base. Preserve it verbatim everywhere you are not grafting.
- For each transplant, place it at the location indicated by GRAFT INTO. If the target is ambiguous or the transplant would not actually improve the passage, SKIP that transplant.
- When grafting, adjust tense, pronoun, point of view, verb agreement, and surrounding punctuation so the seam is invisible. You may add or cut a small number of connective words on either side of the graft if needed for flow. Do not otherwise rewrite the winner's prose.
- Do not invent new content. Do not summarize. Do not editorialize.
- Do not introduce AI prose tells: no em-dash showers, no negation-pivot constructions, no "the way/how" observation framing, no aphoristic closers, no dramatic sentence fragments used for weight.
- If a transplant does not fit cleanly (wrong register, redundant with winner, no sensible anchor), omit it rather than force it.

OUTPUT FORMAT:
First, output the full revised chapter as plain prose. No headings, no preamble, no quotation marks around the whole thing.

Then output a single line of three equals signs on its own line: ===

Then, under the heading GRAFT LOG, list each transplant candidate by its number and state one of:
  APPLIED: <one-sentence note on where/how>
  SKIPPED: <one-sentence reason>

Nothing after the graft log."""


def build_graft_payload(winner_text: str, transplants: List[dict]) -> str:
    """Assemble the user-message payload for the graft call."""
    lines: List[str] = []
    lines.append("INSTRUCTIONS")
    lines.append("=" * 70)
    lines.append(GRAFT_INSTRUCTIONS)
    lines.append("")
    lines.append("=" * 70)
    lines.append("WINNING CHAPTER")
    lines.append("=" * 70)
    lines.append(winner_text.strip())
    lines.append("")
    lines.append("=" * 70)
    lines.append("TRANSPLANT CANDIDATES")
    lines.append("=" * 70)
    for i, tp in enumerate(transplants, start=1):
        lines.append(f"{i}. FROM DRAFT {tp.get('source_draft', '?')} (run {tp.get('source_run_id', '')})")
        lines.append(f'   QUOTE: "{tp.get("quote", "").strip()}"')
        if tp.get("graft_into"):
            lines.append(f"   GRAFT INTO: {tp['graft_into']}")
        if tp.get("why"):
            lines.append(f"   WHY: {tp['why']}")
        lines.append("")
    lines.append("Produce the revised chapter followed by the === separator and the GRAFT LOG, per the instructions above.")
    return "\n".join(lines)


def graft_transplants_with_anthropic(
    api_key: str,
    model: str,
    winner_text: str,
    transplants: List[dict],
) -> dict:
    """Send the winner + transplants to Claude and get back an integrated chapter.

    Returns a dict with keys: grafted_text, graft_log, raw_text, model, stop_reason.
    Uses call_anthropic_with_continuation so long chapters aren't truncated.
    """
    if anthropic is None:
        raise RuntimeError("anthropic package is not installed.")
    if not transplants:
        raise RuntimeError("No transplants to graft.")

    payload = build_graft_payload(winner_text, transplants)

    result = call_anthropic_with_continuation(
        api_key=api_key,
        model=model.strip(),
        payload=payload,
        max_tokens=8000,
        temperature=0.2,
    )

    raw_text = (result.get("text") or "").strip()
    if not raw_text:
        raise RuntimeError("Graft model returned empty text.")

    # Split on the === separator. Be tolerant of extra whitespace.
    separator_match = re.search(r"\n\s*={3,}\s*\n", raw_text)
    if separator_match:
        grafted_text = raw_text[: separator_match.start()].strip()
        remainder = raw_text[separator_match.end():].strip()
        graft_log = re.sub(r"^\s*GRAFT\s*LOG\s*:?\s*\n", "", remainder, flags=re.IGNORECASE).strip()
    else:
        grafted_text = raw_text
        graft_log = "(no graft log parsed — model did not emit the === separator)"

    return {
        "grafted_text": grafted_text,
        "graft_log": graft_log,
        "raw_text": raw_text,
        "model": model.strip(),
        "stop_reason": result.get("stop_reason", ""),
    }


def gather_paths_for_records(df: pd.DataFrame, columns: Iterable[str]) -> List[Path]:
    paths: List[Path] = []
    seen = set()
    for col in columns:
        if col not in df.columns:
            continue
        for raw in df[col].dropna().tolist():
            path = Path(str(raw))
            if path.exists() and path.is_file():
                resolved = str(path.resolve())
                if resolved not in seen:
                    seen.add(resolved)
                    paths.append(path)
    return paths


def export_zip(df: pd.DataFrame, file_paths: List[Path]) -> bytes:
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("results.csv", df.to_csv(index=False))
        for file_path in sorted(file_paths, key=lambda p: p.name):
            zf.write(file_path, arcname=file_path.name)
    mem.seek(0)
    return mem.read()


def make_file_stub(session_id: str, batch_label: str, prompt_id: int, repetition_index: int) -> str:
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_batch = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in batch_label.strip()) or "batch"
    return f"{session_id}_{timestamp_str}_{safe_batch}_p{prompt_id:02d}_r{repetition_index:02d}"


def short_model_slug(model: str) -> str:
    if not model:
        return "model"
    name = model.strip().lower()
    family_map = [
        ("opus", "O"),
        ("sonnet", "S"),
        ("haiku", "H"),
    ]
    prefix = None
    for keyword, letter in family_map:
        if keyword in name:
            prefix = letter
            break
    if prefix is None:
        safe = "".join(ch for ch in model if ch.isalnum() or ch in "-._")
        return safe or "model"
    match = re.search(r"(\d+)[-.](\d+)", name)
    if match:
        return f"{prefix}{match.group(1)}.{match.group(2)}"
    match = re.search(r"\d+", name)
    if match:
        return f"{prefix}{match.group(0)}"
    return prefix


def make_winner_filename(prompt_id: int, temperature: float, model: str) -> str:
    temp_str = f"{temperature:.1f}".lstrip("0") if temperature < 1 else f"{temperature:.1f}"
    safe_temp = temp_str if temp_str.startswith(".") or temp_str.startswith("-") else temp_str
    slug = short_model_slug(model)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"P{prompt_id} T{safe_temp} {slug} Winner {timestamp_str}.txt"


def update_records_bulk(csv_path: Path, run_ids: List[str], updates: dict) -> None:
    df = load_records(csv_path)
    if df.empty or not run_ids:
        return
    mask = df["run_id"].astype(str).isin([str(r) for r in run_ids])
    if not mask.any():
        return
    for key, value in updates.items():
        if key in df.columns:
            if isinstance(value, str):
                df[key] = df[key].astype("object")
            df.loc[mask, key] = value
    df.to_csv(csv_path, index=False)


def coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def clean_api_key(value: str) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", "", str(value)).strip()


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    session_id = ensure_session_state()
    st.title(APP_TITLE)
    st.caption("Run a controlled micro-prompt experiment against one fixed writing package.")

    csv_path = DATA_DIR / "runs.csv"

    # GitHub sync: best-effort. If configured, pull the repo contents down on
    # first render when the local CSV is empty. Failures are non-fatal.
    github_cfg = load_github_config()
    github_pull_on_startup_if_needed(github_cfg, csv_path)

    try:
        prompt_defs = load_prompt_definitions(PROMPTS_CSV)
    except Exception as exc:
        st.error(f"Could not load prompt definitions from {PROMPTS_CSV}: {exc}")
        st.stop()

    with st.sidebar:
        st.header("Run setup")
        st.caption(f"Current app session: {session_id}")
        provider = st.selectbox("Provider", ["anthropic"], index=0)
        model = st.text_input("Model", value=DEFAULT_MODEL)

        api_key = ""
        api_key_source = ""

        try:
            if "ANTHROPIC_API_KEY" in st.secrets:
                api_key = clean_api_key(st.secrets["ANTHROPIC_API_KEY"])
                if api_key:
                    api_key_source = "Streamlit secrets"
        except Exception:
            api_key = ""
            api_key_source = ""

        if not api_key:
            api_key = clean_api_key(os.environ.get("ANTHROPIC_API_KEY", ""))
            if api_key:
                api_key_source = "environment variable"

        if not api_key:
            manual_key = st.text_input(
                "Anthropic API key",
                value="",
                type="password",
                help="Used only for this session if not found in Streamlit secrets or the environment.",
            )
            api_key = clean_api_key(manual_key)
            if api_key:
                api_key_source = "manual entry"

        if api_key:
            st.caption(f"API key loaded from {api_key_source}.")
        else:
            st.error(
                "API key not found. Set ANTHROPIC_API_KEY in Streamlit secrets, "
                "the environment, or enter it manually here."
            )

        temperatures_raw = st.text_input(
            "Temperature(s)",
            value="1.0",
            help="One or more temperatures, comma-separated (e.g. '0.7, 1.0, 1.3'). Each selected prompt will be run at every listed temperature, for the number of repetitions set below.",
        )
        try:
            temperatures = [float(t.strip()) for t in temperatures_raw.split(",") if t.strip()]
            for t in temperatures:
                if t < 0.0 or t > 1.5:
                    raise ValueError(f"Temperature {t} is outside the allowed range 0.0-1.5.")
            if not temperatures:
                raise ValueError("No temperatures provided.")
        except Exception as exc:
            st.error(f"Invalid temperature list: {exc}")
            temperatures = [1.0]
        max_tokens = st.number_input(
            "Max output tokens per API call",
            min_value=500,
            max_value=MAX_ALLOWED_TOKENS,
            value=DEFAULT_MAX_TOKENS,
            step=500,
            help="If the model hits this ceiling, the app will automatically ask it to continue.",
        )
        runs_per_prompt = st.number_input(
            "Runs per prompt",
            min_value=1,
            max_value=10,
            value=1,
            step=1,
            help="Repeat each selected prompt up to 10 times in the same batch.",
        )
        batch_label = st.text_input(
            "Batch label",
            value="batch1",
            help="Optional for your own reference. Current-session export does not depend on this.",
        )
        evaluator_model = st.text_input(
            "Evaluator model",
            value=DEFAULT_EVALUATOR_MODEL,
            help="Claude model used by the 'Evaluate latest batch' button to pick the best draft.",
        )

        if max_tokens < 8000:
            st.warning("This token ceiling is on the low side for full chapter generation. The app can continue automatically, but larger per-call limits are safer.")

        st.markdown("---")
        st.subheader("Storage controls")
        show_current_only = st.checkbox("Show current app session runs only", value=True)
        if st.button("Start a fresh app session", help="Creates a new session ID so new exports include only future runs."):
            st.session_state[CURRENT_SESSION_KEY] = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.session_state[CURRENT_SESSION_RUN_IDS_KEY] = []
            st.rerun()
        if st.button("Clear all stored runs and files", type="secondary", help="Deletes runs.csv and every file in micro_prompt_runs/flat_outputs."):
            clear_all_run_data(csv_path, OUTPUTS_DIR)
            st.success("All stored run data was deleted. A new app session has started.")
            st.rerun()

        st.markdown("---")
        st.subheader("GitHub sync")
        if github_cfg["configured"]:
            st.caption(
                f"Repo: `{github_cfg['repo']}` @ `{github_cfg['branch']}`  \n"
                f"Credentials from: {github_cfg['source']}"
            )
            status = st.session_state.get(GITHUB_SYNC_STATUS_KEY)
            if status:
                msg = f"Last sync {status['timestamp']}: {status['message']}"
                kind = status.get("kind", "info")
                if kind == "error":
                    st.error(msg)
                elif kind == "warn":
                    st.warning(msg)
                elif kind == "success":
                    st.success(msg)
                else:
                    st.info(msg)
            else:
                st.caption("No sync activity yet this session.")

            col_pull, col_push = st.columns(2)
            with col_pull:
                if st.button("Pull from repo", help="Overwrite local files with the repo contents."):
                    with st.spinner("Pulling from GitHub..."):
                        github_pull_all(github_cfg)
                    st.rerun()
            with col_push:
                if st.button("Push all local", help="Push every local file to the repo (best-effort)."):
                    with st.spinner("Pushing to GitHub..."):
                        all_local = [p for p in OUTPUTS_DIR.rglob("*") if p.is_file()]
                        if csv_path.exists():
                            all_local.insert(0, csv_path)
                        github_push_paths(github_cfg, all_local, commit_prefix="manual push")
                    st.rerun()
        else:
            st.caption(
                "Not configured. To enable cross-device continuity, set "
                "`GITHUB_TOKEN` and `GITHUB_REPO` (e.g. `user/chapters`) in "
                "Streamlit secrets. `GITHUB_BRANCH` defaults to `main`."
            )

    left, right = st.columns([1.15, 0.85])

    with left:
        st.subheader("Source package")
        base_prompt = st.text_area("Base prompt", value=DEFAULT_BASE_PROMPT, height=220)

        source_text = ""
        outline_text = ""
        profiles_text = ""
        source_name = ""
        outline_name = ""
        profiles_name = ""

        uploaded_source = st.file_uploader("Upload combined source texts (.txt/.md)", type=["txt", "md"], key="src")
        if uploaded_source is not None:
            source_name = uploaded_source.name
            source_text = decode_uploaded_text(uploaded_source)
            st.info(f"Loaded source text: {source_name}")

        uploaded_outline = st.file_uploader("Upload outline (.txt/.md)", type=["txt", "md"], key="out")
        if uploaded_outline is not None:
            outline_name = uploaded_outline.name
            outline_text = decode_uploaded_text(uploaded_outline)
            st.info(f"Loaded outline: {outline_name}")

        uploaded_profiles = st.file_uploader("Upload character profiles (.txt/.md, optional)", type=["txt", "md"], key="prof")
        if uploaded_profiles is not None:
            profiles_name = uploaded_profiles.name
            profiles_text = decode_uploaded_text(uploaded_profiles)
            st.info(f"Loaded profiles: {profiles_name}")

        st.markdown("### Prompt set")
        df_prompts = pd.DataFrame(prompt_defs)
        st.dataframe(df_prompts, use_container_width=True, hide_index=True)

        selected_ids = st.multiselect(
            "Select prompt IDs to run",
            options=[p["id"] for p in prompt_defs],
            default=[1, 2, 6, 10, 14, 16, 19, 21, 28],
        )

        run_selected = st.button("Run selected prompts", type="primary")

        if run_selected:
            if not api_key:
                st.error("Enter an API key.")
            elif not base_prompt.strip():
                st.error("Base prompt cannot be empty.")
            elif not source_text.strip():
                st.error("Combined source texts are required.")
            elif not outline_text.strip():
                st.error("Outline is required.")
            elif not selected_ids:
                st.error("Select at least one prompt ID.")
            else:
                session_id = ensure_session_state()
                selected_prompts = [p for p in prompt_defs if p["id"] in selected_ids]
                st.session_state[LATEST_BATCH_RUN_IDS_KEY] = []
                progress = st.progress(0)
                status = st.empty()
                failures = []
                warnings = []
                successes = 0
                total_runs = len(selected_prompts) * int(runs_per_prompt) * len(temperatures)
                completed_runs = 0

                for prompt_position, prompt_obj in enumerate(selected_prompts, start=1):
                    payload = build_payload(
                        base_prompt=base_prompt,
                        micro_prompt=prompt_obj["text"],
                        source_text=source_text,
                        outline_text=outline_text,
                        profiles_text=profiles_text,
                    )

                    for temperature in temperatures:
                      for repetition_index in range(1, int(runs_per_prompt) + 1):
                        output_temp_str = f"{float(temperature):.1f}"
                        temp_tag = f"t{output_temp_str.replace('.', 'p')}"
                        file_stub = make_file_stub(session_id, batch_label, prompt_obj["id"], repetition_index) + f"_{temp_tag}"
                        run_id = file_stub

                        output_slug = short_model_slug(model)
                        output_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_filename = (
                            f"P{prompt_obj['id']} T{output_temp_str} {output_slug} "
                            f"R{repetition_index:02d} {output_ts}.txt"
                        )

                        payload_path = OUTPUTS_DIR / f"{file_stub}_payload.txt"
                        output_path = OUTPUTS_DIR / output_filename
                        micro_prompt_path = OUTPUTS_DIR / f"{file_stub}_prompt.txt"
                        meta_path = OUTPUTS_DIR / f"{file_stub}_meta.json"

                        try:
                            status.write(
                                f"Running prompt {prompt_obj['id']} temp {output_temp_str} rep {repetition_index}/{int(runs_per_prompt)} "
                                f"(prompt {prompt_position}/{len(selected_prompts)}, overall {completed_runs + 1}/{total_runs})..."
                            )

                            save_text(payload_path, payload)
                            save_text(micro_prompt_path, prompt_obj["text"])

                            generation = call_anthropic_with_continuation(
                                api_key=api_key,
                                model=model,
                                payload=payload,
                                max_tokens=int(max_tokens),
                                temperature=float(temperature),
                            )

                            output_text = generation["text"]
                            save_text(output_path, output_text)

                            output_hash = sha256_text(output_text)

                            meta = {
                                "run_id": run_id,
                                "session_id": session_id,
                                "timestamp": datetime.now().isoformat(timespec="seconds"),
                                "batch_label": batch_label,
                                "prompt_id": prompt_obj["id"],
                                "repetition_index": repetition_index,
                                "category": prompt_obj["category"],
                                "provider": provider,
                                "model": model,
                                "temperature": float(temperature),
                                "max_tokens": int(max_tokens),
                                "continuation_rounds": generation["continuation_rounds"],
                                "source_name": source_name,
                                "outline_name": outline_name,
                                "profiles_name": profiles_name,
                                "file_stub": file_stub,
                                "payload_file": str(payload_path),
                                "output_file": str(output_path),
                                "micro_prompt_file": str(micro_prompt_path),
                                "meta_file": str(meta_path),
                                "output_sha256": output_hash,
                                "stop_reason": generation["stop_reason"],
                                "input_tokens": generation["input_tokens"],
                                "output_tokens": generation["output_tokens"],
                                "output_words": generation["output_words"],
                                "truncation_flag": generation["truncation_flag"],
                            }
                            save_text(meta_path, json.dumps(meta, indent=2))

                            append_record(
                                csv_path,
                                RunRecord(
                                    run_id=run_id,
                                    session_id=session_id,
                                    timestamp=meta["timestamp"],
                                    batch_label=batch_label,
                                    prompt_id=prompt_obj["id"],
                                    repetition_index=repetition_index,
                                    category=prompt_obj["category"],
                                    provider=provider,
                                    model=model,
                                    temperature=float(temperature),
                                    max_tokens=int(max_tokens),
                                    continuation_rounds=int(generation["continuation_rounds"]),
                                    source_name=source_name,
                                    outline_name=outline_name,
                                    profiles_name=profiles_name,
                                    file_stub=file_stub,
                                    output_file=str(output_path),
                                    payload_file=str(payload_path),
                                    micro_prompt_file=str(micro_prompt_path),
                                    meta_file=str(meta_path),
                                    output_sha256=output_hash,
                                    stop_reason=str(generation["stop_reason"]),
                                    input_tokens=generation["input_tokens"],
                                    output_tokens=generation["output_tokens"],
                                    output_words=generation["output_words"],
                                    truncation_flag=bool(generation["truncation_flag"]),
                                ),
                            )

                            st.session_state[CURRENT_SESSION_RUN_IDS_KEY].append(run_id)
                            st.session_state[LATEST_BATCH_RUN_IDS_KEY].append(run_id)
                            successes += 1

                            # Best-effort push to GitHub. Never blocks on failure.
                            if github_cfg["configured"]:
                                try:
                                    github_push_after_generation(
                                        github_cfg,
                                        csv_path=csv_path,
                                        output_path=output_path,
                                        payload_path=payload_path,
                                        micro_prompt_path=micro_prompt_path,
                                        meta_path=meta_path,
                                    )
                                except Exception as push_exc:
                                    warnings.append(f"GitHub push failed: {push_exc}")

                        except Exception as exc:
                            failures.append(
                                f"Prompt {prompt_obj['id']} temp {output_temp_str} rep {repetition_index}: {exc}"
                            )

                        completed_runs += 1
                        progress.progress(min(completed_runs / total_runs, 1.0))
                        time.sleep(1.25)

                status.write("Run complete.")

                if successes:
                    st.success(f"Completed {successes} run(s).")
                if warnings:
                    st.warning("\n".join(warnings))
                if failures:
                    st.error("\n".join(failures))

    with right:
        st.subheader("Run log")

        df = load_records(csv_path)
        session_run_ids = list(st.session_state.get(CURRENT_SESSION_RUN_IDS_KEY, []))

        if df.empty:
            st.info("No runs logged yet.")
        else:
            display_df = df.copy()
            if "truncation_flag" in display_df.columns:
                display_df["truncation_flag"] = display_df["truncation_flag"].apply(coerce_bool)

            session_df = display_df[display_df["session_id"].astype(str) == session_id].copy() if "session_id" in display_df.columns else display_df.iloc[0:0].copy()
            if session_run_ids:
                session_df = display_df[display_df["run_id"].astype(str).isin(session_run_ids)].copy()

            table_df = session_df if show_current_only else display_df
            if table_df.empty and show_current_only:
                st.info("No runs yet in the current app session.")
            else:
                sorted_table = table_df.sort_values("timestamp", ascending=False).copy()
                if "evaluation_rank" in sorted_table.columns:
                    sorted_table.insert(
                        0,
                        "rank",
                        sorted_table["evaluation_rank"].apply(
                            lambda v: "" if pd.isna(v) else str(int(v))
                        ),
                    )
                if "is_winner" in sorted_table.columns:
                    sorted_table.insert(
                        0,
                        "★",
                        sorted_table["is_winner"].apply(lambda v: "★" if coerce_bool(v) else ""),
                    )
                st.dataframe(sorted_table, use_container_width=True, hide_index=True)

            selectable_df = table_df if not table_df.empty else display_df
            selected_run = st.selectbox("Select run", selectable_df["run_id"].astype(str).tolist())
            current = df[df["run_id"].astype(str) == str(selected_run)].iloc[0]

            with st.form("score_form"):
                originality_label = st.text_input(
                    "Originality label",
                    value=str(current.get("originality_label", "") or ""),
                )
                originality_score = st.text_input(
                    "Originality score",
                    value="" if pd.isna(current.get("originality_score")) else str(current.get("originality_score")),
                )
                manual_rating = st.selectbox(
                    "Manual rating",
                    ["", "strong", "decent", "weak"],
                    index=["", "strong", "decent", "weak"].index(str(current.get("manual_rating", "") or ""))
                    if str(current.get("manual_rating", "") or "") in ["", "strong", "decent", "weak"]
                    else 0,
                )
                manual_notes = st.text_area(
                    "Manual notes",
                    value=str(current.get("manual_notes", "") or ""),
                    height=120,
                )
                submitted = st.form_submit_button("Save score")
                if submitted:
                    parsed_score = None
                    raw = originality_score.strip()
                    if raw:
                        parsed_score = float(raw)
                    update_record(
                        csv_path,
                        selected_run,
                        {
                            "originality_label": originality_label,
                            "originality_score": parsed_score,
                            "manual_rating": manual_rating,
                            "manual_notes": manual_notes,
                        },
                    )
                    st.success("Saved.")
                    st.rerun()

            for label, col in [("Output", "output_file"), ("Micro-prompt", "micro_prompt_file"), ("Payload", "payload_file")]:
                path_str = str(current.get(col, "") or "")
                if path_str and Path(path_str).exists():
                    st.markdown(f"### {label}")
                    content = Path(path_str).read_text(encoding="utf-8")
                    st.text_area(f"{label} preview", value=content, height=320, key=f"preview_{label}_{selected_run}")

            st.markdown("### Selected run metadata")
            st.json({
                "run_id": str(current.get("run_id", "")),
                "session_id": str(current.get("session_id", "")),
                "batch_label": str(current.get("batch_label", "")),
                "prompt_id": int(current.get("prompt_id", 0)),
                "repetition_index": int(current.get("repetition_index", 0)) if not pd.isna(current.get("repetition_index", 0)) else 0,
                "category": str(current.get("category", "")),
                "file_stub": str(current.get("file_stub", "")),
                "stop_reason": str(current.get("stop_reason", "")),
                "output_words": None if pd.isna(current.get("output_words")) else int(current.get("output_words")),
                "continuation_rounds": None if pd.isna(current.get("continuation_rounds")) else int(current.get("continuation_rounds")),
                "truncation_flag": coerce_bool(current.get("truncation_flag")),
                "is_winner": coerce_bool(current.get("is_winner")),
                "evaluation_id": str(current.get("evaluation_id", "") or ""),
                "evaluator_model": str(current.get("evaluator_model", "") or ""),
                "evaluation_parse_status": str(current.get("evaluation_parse_status", "") or ""),
                "evaluation_raw": str(current.get("evaluation_raw", "") or ""),
                "output_sha256": str(current.get("output_sha256", "")),
            })

            history_file_paths = gather_paths_for_records(display_df, ["output_file", "payload_file", "micro_prompt_file", "meta_file"])
            history_zip_bytes = export_zip(display_df, history_file_paths)

            latest_batch_ids = list(st.session_state.get(LATEST_BATCH_RUN_IDS_KEY, []))
            batch_count = len(latest_batch_ids)
            evaluate_clicked = st.button(
                f"Evaluate latest batch ({batch_count} draft{'s' if batch_count != 1 else ''})",
                disabled=batch_count < 2,
                help="Send all drafts from the most recent 'Run selected prompts' click to the evaluator. Opus picks the strongest on literary grounds.",
            )

            if evaluate_clicked:
                if not api_key:
                    st.error("Enter an API key in the sidebar.")
                else:
                    eval_df = load_records(csv_path)
                    batch_rows = eval_df[eval_df["run_id"].astype(str).isin([str(r) for r in latest_batch_ids])].copy()

                    if len(batch_rows) < 2:
                        st.error("Need at least 2 drafts in the latest batch to evaluate.")
                    else:
                        drafts: List[Tuple[str, str]] = []
                        missing: List[str] = []
                        for _, row in batch_rows.iterrows():
                            run_id_val = str(row["run_id"])
                            output_file_val = str(row.get("output_file", "") or "")
                            if output_file_val and Path(output_file_val).exists():
                                draft_text = Path(output_file_val).read_text(encoding="utf-8")
                                drafts.append((run_id_val, draft_text))
                            else:
                                missing.append(run_id_val)

                        if missing:
                            st.warning(f"Skipping {len(missing)} run(s) with missing output files: {', '.join(missing)}")

                        if len(drafts) < 2:
                            st.error("Fewer than 2 drafts have readable output files. Cannot evaluate.")
                        else:
                            with st.spinner(f"Evaluating {len(drafts)} drafts with {evaluator_model}..."):
                                try:
                                    result = evaluate_drafts_with_anthropic(
                                        api_key=api_key,
                                        model=evaluator_model,
                                        drafts=drafts,
                                    )

                                    winner_run_id = result["winner_run_id"]
                                    winner_row = batch_rows[batch_rows["run_id"].astype(str) == str(winner_run_id)].iloc[0]
                                    winner_prompt_id = int(winner_row["prompt_id"])
                                    winner_temperature = float(winner_row["temperature"])
                                    winner_model = str(winner_row["model"])
                                    winner_output_file = str(winner_row["output_file"])

                                    winner_text = Path(winner_output_file).read_text(encoding="utf-8")
                                    winner_filename = make_winner_filename(
                                        prompt_id=winner_prompt_id,
                                        temperature=winner_temperature,
                                        model=winner_model,
                                    )
                                    winner_path = OUTPUTS_DIR / winner_filename
                                    save_text(winner_path, winner_text)

                                    evaluation_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

                                    batch_run_id_list = [str(r) for r in batch_rows["run_id"].astype(str).tolist()]
                                    update_records_bulk(
                                        csv_path,
                                        batch_run_id_list,
                                        {
                                            "is_winner": False,
                                            "evaluation_id": evaluation_id,
                                            "evaluator_model": result["model"],
                                            "evaluation_parse_status": result["parse_status"],
                                            "evaluation_raw": result["raw_text"],
                                            "evaluation_rank": None,
                                        },
                                    )
                                    update_record(
                                        csv_path,
                                        str(winner_run_id),
                                        {"is_winner": True},
                                    )

                                    # Per-run rank: ranking[0] is draft #1's 1-indexed position, etc.
                                    # ranking is a list of draft indices (1-based) ordered best-to-worst.
                                    ranking = result.get("ranking") or []
                                    run_id_by_draft_index = {
                                        idx + 1: drafts[idx][0] for idx in range(len(drafts))
                                    }
                                    ranked_rows: List[Tuple[str, int]] = []
                                    for rank_position, draft_number in enumerate(ranking, start=1):
                                        run_id_for_rank = run_id_by_draft_index.get(draft_number)
                                        if run_id_for_rank is None:
                                            continue
                                        update_record(
                                            csv_path,
                                            str(run_id_for_rank),
                                            {"evaluation_rank": rank_position},
                                        )
                                        ranked_rows.append((str(run_id_for_rank), rank_position))

                                    # Build and save a transplants report alongside the winner.
                                    transplants = result.get("transplants") or []
                                    transplants_path: Optional[Path] = None
                                    if transplants:
                                        report_lines = [
                                            f"Transplant candidates for winner: {winner_filename}",
                                            f"Winner run_id: {winner_run_id}",
                                            f"Evaluator: {result['model']}",
                                            f"Evaluation id: {evaluation_id}",
                                            "",
                                            "=" * 70,
                                            "",
                                        ]
                                        for i, tp in enumerate(transplants, start=1):
                                            src_run = tp.get("source_run_id", "")
                                            report_lines.append(f"{i}. FROM DRAFT {tp['source_draft']} (run {src_run})")
                                            report_lines.append(f'   QUOTE: "{tp["quote"]}"')
                                            if tp.get("graft_into"):
                                                report_lines.append(f"   GRAFT INTO: {tp['graft_into']}")
                                            if tp.get("why"):
                                                report_lines.append(f"   WHY: {tp['why']}")
                                            report_lines.append("")
                                        transplants_filename = winner_filename.rsplit(".txt", 1)[0] + " Transplants.txt"
                                        transplants_path = OUTPUTS_DIR / transplants_filename
                                        save_text(transplants_path, "\n".join(report_lines))

                                    st.session_state["last_evaluation_result"] = {
                                        "evaluation_id": evaluation_id,
                                        "winner_run_id": str(winner_run_id),
                                        "winner_index": result["winner_index"],
                                        "total_drafts": len(drafts),
                                        "parse_status": result["parse_status"],
                                        "raw_text": result["raw_text"],
                                        "winner_filename": winner_filename,
                                        "evaluator_model": result["model"],
                                        "batch_run_ids": batch_run_id_list,
                                        "ranking": ranking,
                                        "ranked_rows": ranked_rows,
                                        "transplants": transplants,
                                        "transplants_file": str(transplants_path) if transplants_path else "",
                                    }

                                    st.success(
                                        f"Winner: {winner_run_id} (draft {result['winner_index']} of {len(drafts)}). "
                                        f"Parse: {result['parse_status']}. Saved to {winner_filename}."
                                    )
                                    if result["parse_status"] != "clean":
                                        st.info(f"Raw evaluator response: {result['raw_text']!r}")

                                    # Best-effort push of evaluation artifacts to GitHub.
                                    if github_cfg["configured"]:
                                        try:
                                            github_push_after_evaluation(
                                                github_cfg,
                                                csv_path=csv_path,
                                                winner_path=winner_path,
                                                transplants_path=transplants_path,
                                            )
                                        except Exception as push_exc:
                                            st.warning(f"GitHub push failed: {push_exc}")

                                except Exception as eval_exc:
                                    st.error(f"Evaluation failed: {eval_exc}")

            # Persistent panel showing the most recent evaluation's full reasoning.
            last_eval = st.session_state.get("last_evaluation_result")
            if last_eval:
                st.markdown("### Latest evaluation")
                st.markdown(
                    f"**Winner:** draft {last_eval['winner_index']} of {last_eval['total_drafts']} "
                    f"— run `{last_eval['winner_run_id']}`  \n"
                    f"**Saved as:** `{last_eval['winner_filename']}`  \n"
                    f"**Evaluator:** `{last_eval['evaluator_model']}` "
                    f"(parse: {last_eval['parse_status']})"
                )

                ranked_rows = last_eval.get("ranked_rows") or []
                transplants = last_eval.get("transplants") or []

                if ranked_rows:
                    rank_df = pd.DataFrame(
                        [{"rank": rank, "run_id": rid} for rid, rank in ranked_rows]
                    )
                    rank_df = rank_df.sort_values("rank").reset_index(drop=True)
                    st.markdown("**Ranking (best to worst):**")
                    st.dataframe(rank_df, use_container_width=True, hide_index=True)
                else:
                    st.caption("No ranking line was returned by the evaluator.")

                if transplants:
                    st.markdown(f"**Transplant candidates ({len(transplants)}):**")
                    st.caption(
                        "Lines from losing drafts the evaluator thinks could be grafted into the winner."
                    )
                    for i, tp in enumerate(transplants, start=1):
                        src_run = tp.get("source_run_id", "")
                        header = f"**{i}. From draft {tp['source_draft']}**"
                        if src_run:
                            header += f" — run `{src_run}`"
                        st.markdown(header)
                        st.markdown(f"> {tp['quote']}")
                        if tp.get("graft_into"):
                            st.markdown(f"*Graft into:* {tp['graft_into']}")
                        if tp.get("why"):
                            st.markdown(f"*Why:* {tp['why']}")
                        st.markdown("")
                    tp_file = last_eval.get("transplants_file") or ""
                    if tp_file and Path(tp_file).exists():
                        st.caption(f"Saved to: `{Path(tp_file).name}`")

                    # --- Graft transplants button -------------------------------------
                    st.markdown("---")
                    winner_filename_for_graft = last_eval.get("winner_filename") or ""
                    winner_path_for_graft = OUTPUTS_DIR / winner_filename_for_graft if winner_filename_for_graft else None
                    graft_ready = bool(
                        winner_path_for_graft
                        and winner_path_for_graft.exists()
                        and api_key
                    )

                    graft_col_a, graft_col_b = st.columns([1, 2])
                    with graft_col_a:
                        graft_clicked = st.button(
                            "Graft transplants into winner",
                            disabled=not graft_ready,
                            help=(
                                "Sends the winner and the transplant list to Claude, which integrates "
                                "each transplant at its target location, adjusts tense/POV/flow at the "
                                "seam, and skips any transplant that does not fit cleanly. Saves the "
                                "result as a new file you can download."
                            ),
                        )
                    with graft_col_b:
                        if not api_key:
                            st.caption("Set your Anthropic API key to enable grafting.")
                        elif not (winner_path_for_graft and winner_path_for_graft.exists()):
                            st.caption("Winner file not found on disk.")

                    if graft_clicked and graft_ready:
                        try:
                            winner_text_for_graft = winner_path_for_graft.read_text(encoding="utf-8")
                            with st.spinner(f"Grafting {len(transplants)} transplant(s) with {evaluator_model}..."):
                                graft_result = graft_transplants_with_anthropic(
                                    api_key=api_key,
                                    model=evaluator_model,
                                    winner_text=winner_text_for_graft,
                                    transplants=transplants,
                                )

                            grafted_stub = winner_filename_for_graft.rsplit(".txt", 1)[0]
                            grafted_filename = f"{grafted_stub} Grafted.txt"
                            grafted_path = OUTPUTS_DIR / grafted_filename
                            save_text(grafted_path, graft_result["grafted_text"])

                            log_filename = f"{grafted_stub} Graft Log.txt"
                            log_path = OUTPUTS_DIR / log_filename
                            log_body = (
                                f"Graft log for: {grafted_filename}\n"
                                f"Source winner: {winner_filename_for_graft}\n"
                                f"Model: {graft_result['model']}\n"
                                f"Stop reason: {graft_result.get('stop_reason', '')}\n"
                                f"{'=' * 70}\n\n"
                                f"{graft_result['graft_log']}\n"
                            )
                            save_text(log_path, log_body)

                            st.session_state["last_graft_result"] = {
                                "grafted_filename": grafted_filename,
                                "grafted_path": str(grafted_path),
                                "log_filename": log_filename,
                                "log_path": str(log_path),
                                "graft_log": graft_result["graft_log"],
                                "stop_reason": graft_result.get("stop_reason", ""),
                                "source_winner": winner_filename_for_graft,
                            }

                            if github_cfg["configured"]:
                                try:
                                    github_push_paths(
                                        github_cfg,
                                        [grafted_path, log_path],
                                        commit_prefix="graft",
                                    )
                                except Exception as push_exc:
                                    st.warning(f"Graft saved locally; GitHub push failed: {push_exc}")

                            st.success(f"Grafted chapter saved as `{grafted_filename}`.")
                        except Exception as graft_exc:
                            st.error(f"Graft failed: {graft_exc}")

                    last_graft = st.session_state.get("last_graft_result")
                    if (
                        last_graft
                        and last_graft.get("source_winner") == winner_filename_for_graft
                        and Path(last_graft.get("grafted_path", "")).exists()
                    ):
                        grafted_path_obj = Path(last_graft["grafted_path"])
                        grafted_bytes = grafted_path_obj.read_bytes()
                        st.download_button(
                            "Download grafted chapter",
                            data=grafted_bytes,
                            file_name=last_graft["grafted_filename"],
                            mime="text/plain",
                            key=f"dl_graft_{last_graft['grafted_filename']}",
                        )
                        with st.expander("Graft log (what was applied vs. skipped)", expanded=False):
                            st.markdown(last_graft["graft_log"])
                            log_path_obj = Path(last_graft.get("log_path", ""))
                            if log_path_obj.exists():
                                st.caption(f"Saved to: `{log_path_obj.name}`")
                    # ------------------------------------------------------------------
                else:
                    st.caption("No transplant candidates proposed by the evaluator.")

                with st.expander("Evaluator reasoning", expanded=True):
                    st.markdown(last_eval["raw_text"])

            st.download_button(
                "Download all history",
                data=history_zip_bytes,
                file_name="micro_prompt_runs_export_all.zip",
                mime="application/zip",
            )

            current_session_export_df = session_df if not session_df.empty else display_df.iloc[0:0].copy()
            current_session_file_paths = gather_paths_for_records(current_session_export_df, ["output_file", "payload_file", "micro_prompt_file", "meta_file"])
            current_session_zip_bytes = export_zip(current_session_export_df, current_session_file_paths)
            st.download_button(
                "Download current app session only",
                data=current_session_zip_bytes,
                file_name=f"micro_prompt_runs_session_{session_id}.zip",
                mime="application/zip",
                disabled=current_session_export_df.empty,
            )


if __name__ == "__main__":
    main()