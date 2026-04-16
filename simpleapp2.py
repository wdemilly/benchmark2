"""
Micro-Prompt Harness — Simplified
=================================
Generate chapter drafts from prompt variants × temperatures × repetitions.
Evaluate batches with Opus. Pick a winner. Sync to GitHub.

The generation prompt lives in prompts.csv. The app does not inject its own
drafting instructions. Whatever the prompt says, the model gets — plus the
uploaded documents as context.
"""

import base64
import io
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List

import pandas as pd
import requests
import streamlit as st

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import docx as python_docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False


# ============================================================================
# Constants
# ============================================================================

RUNS_DIR = Path("micro_prompt_runs")
OUTPUTS_DIR = RUNS_DIR / "flat_outputs"
CSV_FILENAME = "runs.csv"
PROMPTS_CSV = "prompts.csv"

DEFAULT_GEN_MODEL = "claude-opus-4-6"
DEFAULT_EVAL_MODEL = "claude-sonnet-4-5"
MAX_GEN_TOKENS = 16000
MAX_EVAL_TOKENS = 6000


# ============================================================================
# Evaluator prompt — genre reader, not chapter-specific
# ============================================================================

EVALUATOR_PROMPT = """You are an experienced reader of commercial fiction in this genre. You will read {N} drafts of the same chapter, generated from the same source material.

Read every draft in full. Do not skim. Infer the genre, period, point of view, and voice from the drafts themselves.

Rank them according to the elements of good writing, as a developmental editor and commercial fiction reader would judge them:

- Premise delivery: Does the chapter establish clear stakes and a promise to the reader?
- Plot and scene logic: Does the chapter move with causality and escalation? Do scenes earn their place?
- Characters: Do characters have desire lines, agency, and dimensionality? Can you tell them apart?
- IMPORTANT: Dialogue: Does it carry voice distinction, subtext, tension, and utility — or is it polite, flat, expository, and interchangeable?
- Prose style: Clarity, rhythm, diction, consistency. Does the prose serve the story or perform for its own sake?
- Pacing: Momentum, scene length, transitions. Does the chapter earn its length or does it sag?
- Setting: Atmosphere through specificity and integration with action — not scenic painting.
- Emotional resonance: Tension, payoff, empathy, curiosity, surprise. Does the chapter make you feel something earned?
- Show versus tell: Does the writer trust the reader, or explain what just happened?

Be demanding. Do not be diplomatic. If two drafts are close, name the specific thing that tips the decision.

OUTPUT FORMAT

For each draft, write a brief paragraph (2-4 sentences) citing a specific passage.

Then a comparison paragraph naming the top 2-3 contenders and why the top one edges the others.

Then on a line by itself:

RANKING: N, N, N, ...

(every draft number from strongest to weakest, separated by commas, each draft exactly once)

Then on the final line:

WINNER: N

Nothing after that line."""


# ============================================================================
# Data model
# ============================================================================

@dataclass
class RunRecord:
    run_id: str = ""
    timestamp: str = ""
    prompt_id: int = 0
    prompt_text: str = ""
    temperature: float = 0.7
    model: str = ""
    output_file: str = ""
    payload_file: str = ""
    meta_file: str = ""
    word_count: int = 0
    is_winner: bool = False
    evaluation_id: str = ""
    evaluation_rank: int = 0
    evaluator_model: str = ""
    evaluation_parse_status: str = ""
    evaluation_raw: str = ""


RUN_FIELDS = list(RunRecord.__dataclass_fields__.keys())


# ============================================================================
# File I/O
# ============================================================================

def ensure_dirs():
    RUNS_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)


def save_text(path: Path, text: str):
    path.write_text(text, encoding="utf-8")


def load_csv(path: Path) -> pd.DataFrame:
    if path.exists() and path.stat().st_size > 0:
        df = pd.read_csv(path, dtype=str)
        for col in RUN_FIELDS:
            if col not in df.columns:
                df[col] = ""
        return df
    return pd.DataFrame(columns=RUN_FIELDS)


def append_record(path: Path, record: RunRecord):
    df = load_csv(path)
    new_row = pd.DataFrame([asdict(record)])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(path, index=False)


def update_record(path: Path, run_id: str, updates: dict):
    df = load_csv(path)
    mask = df["run_id"].astype(str) == str(run_id)
    for k, v in updates.items():
        if k in df.columns:
            df[k] = df[k].astype(object)
        df.loc[mask, k] = v
    df.to_csv(path, index=False)


def update_records_bulk(path: Path, run_ids: list, updates: dict):
    df = load_csv(path)
    mask = df["run_id"].astype(str).isin([str(r) for r in run_ids])
    for k, v in updates.items():
        if k in df.columns:
            df[k] = df[k].astype(object)
        df.loc[mask, k] = v
    df.to_csv(path, index=False)


def extract_text_from_upload(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".txt"):
            data = uploaded_file.read()
            uploaded_file.seek(0)
            return data.decode("utf-8", errors="replace") if isinstance(data, bytes) else data
        elif name.endswith(".docx") and DOCX_AVAILABLE:
            doc = python_docx.Document(uploaded_file)
            uploaded_file.seek(0)
            return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        st.warning(f"Could not read {uploaded_file.name}: {e}")
    return ""


# ============================================================================
# API key loading — secrets first, then env, then manual
# ============================================================================

def clean_api_key(value: str) -> str:
    return value.strip().strip("'\"").strip()


def load_api_key() -> tuple[str, str]:
    """Return (api_key, source_label). Checks Streamlit secrets, then env, then empty."""
    try:
        if "ANTHROPIC_API_KEY" in st.secrets:
            key = clean_api_key(str(st.secrets["ANTHROPIC_API_KEY"]))
            if key:
                return key, "Streamlit secrets"
    except Exception:
        pass

    env_key = clean_api_key(os.environ.get("ANTHROPIC_API_KEY", ""))
    if env_key:
        return env_key, "environment variable"

    return "", ""


# ============================================================================
# Prompt loading
# ============================================================================

def load_prompts() -> pd.DataFrame:
    path = Path(PROMPTS_CSV)
    if not path.exists():
        return pd.DataFrame(columns=["id", "text", "category"])
    df = pd.read_csv(path)
    if "id" not in df.columns or "text" not in df.columns:
        st.error(f"{PROMPTS_CSV} must have 'id' and 'text' columns.")
        return pd.DataFrame(columns=["id", "text", "category"])
    if "category" not in df.columns:
        df["category"] = ""
    return df


# ============================================================================
# Payload construction — minimal
# ============================================================================

def build_payload(prompt_text: str, doc_texts: dict[str, str]) -> str:
    parts = [prompt_text.strip()]
    for label, text in doc_texts.items():
        if text.strip():
            parts.append(f"\n\n=== {label.upper()} ===\n\n{text.strip()}")
    parts.append(
        "\n\nWrite the full chapter now. Return plain text only, "
        "with normal paragraph breaks and no commentary."
    )
    return "\n".join(parts)


# ============================================================================
# Generation
# ============================================================================

def generate_chapter(client, model: str, temperature: float, payload: str) -> str:
    resp = client.messages.create(
        model=model,
        max_tokens=MAX_GEN_TOKENS,
        temperature=temperature,
        messages=[{"role": "user", "content": payload}],
    )
    return "\n".join(b.text for b in resp.content if getattr(b, "text", None))


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_drafts_with_anthropic(client, model: str, drafts: list[dict]) -> dict:
    n = len(drafts)
    parts = [EVALUATOR_PROMPT.format(N=n), "\n\n"]
    for i, d in enumerate(drafts, 1):
        parts.append(f"=== DRAFT {i} (run_id: {d['run_id']}) ===\n\n{d['text']}\n\n")

    resp = client.messages.create(
        model=model,
        max_tokens=MAX_EVAL_TOKENS,
        temperature=0,
        messages=[{"role": "user", "content": "".join(parts)}],
    )
    raw = "\n".join(b.text for b in resp.content if getattr(b, "text", None))

    ranking = list(range(1, n + 1))
    parse_status = "clean"
    rank_match = re.search(r"RANKING:\s*([0-9,\s]+)", raw)
    if rank_match:
        nums = [int(x.strip()) for x in rank_match.group(1).split(",") if x.strip().isdigit()]
        seen = set()
        deduped = []
        for x in nums:
            if 1 <= x <= n and x not in seen:
                seen.add(x)
                deduped.append(x)
        missing = [i for i in range(1, n + 1) if i not in seen]
        ranking = deduped + missing
        if missing:
            parse_status = "partial"
    else:
        parse_status = "no_ranking_line"

    winner_match = re.search(r"WINNER:\s*(\d+)", raw)
    if winner_match:
        winner_idx = int(winner_match.group(1))
    else:
        winner_idx = ranking[0] if ranking else 1
        if parse_status == "clean":
            parse_status = "no_winner_line"

    winner_idx = max(1, min(winner_idx, n))
    winner_run_id = drafts[winner_idx - 1]["run_id"]

    return {
        "winner_run_id": winner_run_id,
        "winner_index": winner_idx,
        "ranking": ranking,
        "raw_text": raw,
        "parse_status": parse_status,
        "model": model,
    }


# ============================================================================
# File naming
# ============================================================================

def make_file_stub(prompt_id: int, temperature: float, model: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = model.split("-")[-1][:6] if "-" in model else model[:6]
    return f"P{prompt_id} T{temperature} {model_short} {ts}"


def make_winner_filename(prompt_id: int, temperature: float, model: str) -> str:
    stub = make_file_stub(prompt_id, temperature, model)
    return f"WINNER {stub}.txt"


# ============================================================================
# Export
# ============================================================================

def export_zip(df: pd.DataFrame, file_paths: list[Path]) -> bytes:
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        zf.writestr("runs.csv", csv_buf.getvalue())
        for p in file_paths:
            if p.exists():
                zf.write(p, p.name)
    return buf.getvalue()


def gather_output_paths(df: pd.DataFrame) -> list[Path]:
    paths = []
    for col in ["output_file", "payload_file", "meta_file"]:
        if col in df.columns:
            for val in df[col].dropna():
                p = Path(str(val))
                if p.exists():
                    paths.append(p)
    return paths


# ============================================================================
# GitHub sync — best-effort overlay
# ============================================================================

GITHUB_API_BASE = "https://api.github.com"
GITHUB_SYNC_STATUS_KEY = "github_sync_status"
GITHUB_PULLED_KEY = "github_pulled_this_session"


def load_github_config() -> dict:
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


def _local_path_for_repo_path(repo_path: str) -> Path:
    return RUNS_DIR / repo_path


def _repo_path_for_local(local_path: Path) -> Optional[str]:
    try:
        rel = local_path.resolve().relative_to(RUNS_DIR.resolve())
    except Exception:
        return None
    return rel.as_posix()


def github_list_tree(cfg: dict) -> List[dict]:
    if not cfg.get("configured"):
        return []
    repo = cfg["repo"]
    branch = cfg["branch"]
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
        return []
    if not branch_resp.ok:
        _gh_record_status(f"GitHub list failed: {branch_resp.status_code}", kind="error")
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
        return []
    entries = tree_resp.json().get("tree", []) or []
    return [
        {"path": entry["path"], "sha": entry["sha"]}
        for entry in entries
        if entry.get("type") == "blob" and entry.get("path")
    ]


def github_get_file_bytes(cfg: dict, path: str) -> Optional[bytes]:
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
    if body.get("encoding") == "base64" and "content" in body:
        try:
            return base64.b64decode(body["content"])
        except Exception:
            return None
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
    if not resp.ok:
        return None
    return resp.json().get("sha")


def github_put_file(cfg: dict, path: str, data: bytes, message: str) -> bool:
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


def github_pull_all(cfg: dict) -> dict:
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
    if result["failed"] == 0 and result["pushed"]:
        _gh_record_status(
            f"Pushed {result['pushed']} file(s) to {cfg['repo']}",
            kind="success",
        )
    return result


def github_push_after_generation(
    cfg: dict, csv_path: Path, output_path: Path, payload_path: Path, meta_path: Path,
) -> None:
    if not cfg.get("configured"):
        return
    github_push_paths(
        cfg, [csv_path, output_path, payload_path, meta_path],
        commit_prefix="generation",
    )


def github_push_after_evaluation(
    cfg: dict, csv_path: Path, winner_path: Path,
) -> None:
    if not cfg.get("configured"):
        return
    github_push_paths(cfg, [csv_path, winner_path], commit_prefix="evaluation")


def github_pull_on_startup_if_needed(cfg: dict, csv_path: Path) -> None:
    if not cfg.get("configured"):
        return
    if st.session_state.get(GITHUB_PULLED_KEY):
        return
    local_empty = (not csv_path.exists()) or csv_path.stat().st_size == 0
    if local_empty:
        github_pull_all(cfg)
    st.session_state[GITHUB_PULLED_KEY] = True


# ============================================================================
# Streamlit UI
# ============================================================================

st.set_page_config(page_title="Micro-Prompt Harness", layout="wide")
st.title("Micro-Prompt Harness")
st.caption("Generate · Evaluate · Pick the winner")

ensure_dirs()
csv_path = RUNS_DIR / CSV_FILENAME

# Load configs
github_cfg = load_github_config()
github_pull_on_startup_if_needed(github_cfg, csv_path)

auto_key, auto_key_source = load_api_key()

# --- Sidebar ---
with st.sidebar:
    st.header("Configuration")

    # API key — auto-loaded or manual
    if auto_key:
        api_key = auto_key
        st.success(f"API key loaded from {auto_key_source}")
    else:
        manual_key = st.text_input("Anthropic API Key", type="password")
        api_key = clean_api_key(manual_key) if manual_key else ""
        if not api_key:
            st.warning("Set ANTHROPIC_API_KEY in Streamlit secrets or enter above.")

    st.markdown("---")

    # Model selection
    gen_model = st.text_input("Generation model", value=DEFAULT_GEN_MODEL)
    eval_model = st.text_input("Evaluation model", value=DEFAULT_EVAL_MODEL)

    st.markdown("---")

    # Temperature
    temps_input = st.text_input("Temperatures (comma-separated)", value="0.6, 0.7")
    try:
        temperatures = [float(t.strip()) for t in temps_input.split(",") if t.strip()]
    except ValueError:
        temperatures = [0.7]
        st.warning("Could not parse temperatures. Using 0.7.")

    # Repetitions
    repetitions = st.number_input("Repetitions per prompt×temp", min_value=1, max_value=10, value=3)

    st.markdown("---")

    # Document uploads
    st.subheader("Documents")
    st.caption("Upload the files the prompt references.")

    doc_uploads = {}
    outline_file = st.file_uploader("Outline", type=["txt", "docx"], key="outline")
    if outline_file:
        doc_uploads["Outline"] = extract_text_from_upload(outline_file)

    source_file = st.file_uploader("Source text (voice model)", type=["txt", "docx"], key="source")
    if source_file:
        doc_uploads["Source Text"] = extract_text_from_upload(source_file)

    profiles_file = st.file_uploader("Character profiles", type=["txt", "docx"], key="profiles")
    if profiles_file:
        doc_uploads["Character Profiles"] = extract_text_from_upload(profiles_file)

    st.markdown("---")

    # GitHub sync status
    st.subheader("GitHub sync")
    if github_cfg["configured"]:
        st.success(f"Repo: `{github_cfg['repo']}` ({github_cfg['source']})")
        sync_status = st.session_state.get(GITHUB_SYNC_STATUS_KEY)
        if sync_status:
            st.caption(f"{sync_status['timestamp']}: {sync_status['message']}")
        if st.button("Sync now (pull)"):
            github_pull_all(github_cfg)
            st.rerun()
    else:
        st.info("Set GITHUB_TOKEN and GITHUB_REPO in secrets to enable sync.")

    st.markdown("---")
    st.caption(f"Gen: `{gen_model}` · Eval: `{eval_model}`")
    st.caption(f"Temps: {temperatures} · Reps: {repetitions}")
    if doc_uploads:
        st.caption(f"Docs: {', '.join(doc_uploads.keys())}")

# --- Load prompts ---
prompts_df = load_prompts()

if prompts_df.empty:
    st.warning(
        f"No `{PROMPTS_CSV}` found or it has no rows. "
        f"Create a CSV with columns `id` and `text` (and optionally `category`)."
    )
    st.stop()

# --- Main area: two columns ---
left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("Prompts")

    prompt_options = {
        f"P{row['id']}: {str(row['text'])[:80]}...": int(row["id"])
        for _, row in prompts_df.iterrows()
    }
    selected_labels = st.multiselect(
        "Select prompt(s) to run",
        options=list(prompt_options.keys()),
        default=list(prompt_options.keys()),
    )
    selected_ids = [prompt_options[label] for label in selected_labels]

    for pid in selected_ids:
        row = prompts_df[prompts_df["id"].astype(int) == pid].iloc[0]
        with st.expander(f"P{pid} — {str(row.get('category', ''))}"):
            st.text(str(row["text"]))

    total_runs = len(selected_ids) * len(temperatures) * repetitions
    st.write(
        f"**{len(selected_ids)}** prompts × **{len(temperatures)}** temps × "
        f"**{repetitions}** reps = **{total_runs}** drafts"
    )

    # Generate button
    if st.button("Generate", type="primary", disabled=not api_key or total_runs == 0):
        client = anthropic.Anthropic(api_key=api_key)
        progress = st.progress(0.0)
        status = st.empty()
        run_count = 0

        for pid in selected_ids:
            prompt_row = prompts_df[prompts_df["id"].astype(int) == pid].iloc[0]
            prompt_text = str(prompt_row["text"])

            for temp in temperatures:
                for rep in range(1, repetitions + 1):
                    run_count += 1
                    status.info(f"Run {run_count}/{total_runs}: P{pid} T{temp} R{rep:02d}")

                    payload = build_payload(prompt_text, doc_uploads)
                    stub = make_file_stub(pid, temp, gen_model)
                    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:20]

                    payload_path = OUTPUTS_DIR / f"{stub}_payload.txt"
                    save_text(payload_path, payload)

                    try:
                        output = generate_chapter(client, gen_model, temp, payload)
                    except Exception as e:
                        st.error(f"Generation failed for P{pid} T{temp} R{rep}: {e}")
                        continue

                    output_path = OUTPUTS_DIR / f"{stub}_output.txt"
                    save_text(output_path, output)

                    meta = {
                        "run_id": run_id,
                        "prompt_id": pid,
                        "temperature": temp,
                        "model": gen_model,
                        "repetition": rep,
                        "timestamp": datetime.now().isoformat(),
                        "documents": list(doc_uploads.keys()),
                    }
                    meta_path = OUTPUTS_DIR / f"{stub}_meta.json"
                    save_text(meta_path, json.dumps(meta, indent=2))

                    record = RunRecord(
                        run_id=run_id,
                        timestamp=datetime.now().isoformat(),
                        prompt_id=pid,
                        prompt_text=prompt_text[:200],
                        temperature=temp,
                        model=gen_model,
                        output_file=str(output_path),
                        payload_file=str(payload_path),
                        meta_file=str(meta_path),
                        word_count=len(output.split()),
                    )
                    append_record(csv_path, record)

                    # Push to GitHub
                    if github_cfg["configured"]:
                        try:
                            github_push_after_generation(
                                github_cfg, csv_path, output_path, payload_path, meta_path,
                            )
                        except Exception as push_exc:
                            st.warning(f"GitHub push failed: {push_exc}")

                    progress.progress(run_count / total_runs)
                    time.sleep(0.5)

        progress.empty()
        status.success(f"Done. {run_count} drafts generated.")
        st.rerun()


with right_col:
    st.subheader("Run log")

    df = load_csv(csv_path)
    if df.empty:
        st.info("No runs yet. Generate some drafts.")
    else:
        display_cols = ["run_id", "prompt_id", "temperature", "model", "word_count",
                        "is_winner", "evaluation_rank"]
        available = [c for c in display_cols if c in df.columns]
        st.dataframe(df[available], use_container_width=True)

        # --- Evaluate ---
        st.markdown("---")
        st.subheader("Evaluate")

        max_eval = min(25, len(df))
        eval_count = st.slider("Drafts to evaluate (most recent N)", 2, max_eval, min(12, max_eval))

        if st.button("Evaluate", type="primary", disabled=not api_key):
            client = anthropic.Anthropic(api_key=api_key)
            batch_df = df.tail(eval_count).copy()
            drafts = []
            for _, row in batch_df.iterrows():
                output_path = Path(str(row["output_file"]))
                if output_path.exists():
                    text = output_path.read_text(encoding="utf-8")
                    drafts.append({"run_id": str(row["run_id"]), "text": text})

            if len(drafts) < 2:
                st.error("Need at least 2 readable drafts to evaluate.")
            else:
                with st.spinner(f"Evaluating {len(drafts)} drafts with {eval_model}..."):
                    try:
                        result = evaluate_drafts_with_anthropic(client, eval_model, drafts)

                        winner_run_id = result["winner_run_id"]
                        winner_row = batch_df[batch_df["run_id"].astype(str) == str(winner_run_id)].iloc[0]

                        winner_text = Path(str(winner_row["output_file"])).read_text(encoding="utf-8")
                        winner_filename = make_winner_filename(
                            int(winner_row["prompt_id"]),
                            float(winner_row["temperature"]),
                            str(winner_row["model"]),
                        )
                        winner_path = OUTPUTS_DIR / winner_filename
                        save_text(winner_path, winner_text)

                        evaluation_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        batch_run_ids = [str(r) for r in batch_df["run_id"].astype(str).tolist()]
                        update_records_bulk(csv_path, batch_run_ids, {
                            "is_winner": False,
                            "evaluation_id": evaluation_id,
                            "evaluator_model": result["model"],
                            "evaluation_parse_status": result["parse_status"],
                            "evaluation_raw": result["raw_text"],
                        })
                        update_record(csv_path, str(winner_run_id), {"is_winner": True})

                        for rank_pos, draft_num in enumerate(result["ranking"], 1):
                            rid = drafts[draft_num - 1]["run_id"]
                            update_record(csv_path, rid, {"evaluation_rank": rank_pos})

                        st.success(f"Winner: {winner_run_id}. Saved to {winner_filename}.")

                        st.markdown("**Ranking (best → worst):**")
                        for rank_pos, draft_num in enumerate(result["ranking"], 1):
                            d = drafts[draft_num - 1]
                            marker = " ★" if d["run_id"] == winner_run_id else ""
                            st.write(f"{rank_pos}. {d['run_id']}{marker}")

                        with st.expander("Evaluator reasoning"):
                            st.text(result["raw_text"])

                        # Push evaluation to GitHub
                        if github_cfg["configured"]:
                            try:
                                github_push_after_evaluation(github_cfg, csv_path, winner_path)
                            except Exception as push_exc:
                                st.warning(f"GitHub push failed: {push_exc}")

                    except Exception as e:
                        st.error(f"Evaluation failed: {e}")

                st.rerun()

        # --- Downloads ---
        st.markdown("---")
        all_paths = gather_output_paths(df)
        if all_paths:
            zip_bytes = export_zip(df, all_paths)
            st.download_button(
                "Download all runs (ZIP)",
                data=zip_bytes,
                file_name=f"micro_prompt_runs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip",
            )

        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        st.download_button(
            "Download runs.csv",
            data=csv_buf.getvalue(),
            file_name="runs.csv",
            mime="text/csv",
        )
