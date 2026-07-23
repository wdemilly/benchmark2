"""
step2a_api_test.py -- ROUTE A: the API replication of the Step 2A trial
=======================================================================

This is an OFF-TO-THE-SIDE experiment, not part of the writing pipeline. The
pipeline is prompts pasted by hand into incognito browser windows; this app
exists only to check one thing: does an API assessor choose the same drafts a
by-hand browser window chose? Originality is nowhere in it.

What it does:
  * reads the eight paste-ready bundles already sitting in
    Step2A_RouteB_Test/ (the same files the browser windows used),
  * sends each whole bundle to Claude through the API -- the same streamed
    call shape as simpleapp714.py, adaptive thinking at high effort by
    default, which mirrors the web interface's recipe (ledger Entry 22),
  * reads the "SELECTED: DRAFT n" line out of each reply,
  * scores that pick against the KEY the assembler wrote (which draft holds
    which Originality score), and shows it beside the browser's pick.

IMPORTANT, carried from ledger Entry 55: the Originality score is NOT a
measure of writing quality. A lower-scoring draft can be the better prose.
"Hit the top score" below means only "chose the highest Originality draft";
it is not a quality verdict. The quality verdict needs Walter to read the
drafts. This app measures agreement with the browser and the score pattern,
nothing more.

Run, from the BandPredictor folder that holds this file and the
Step2A_RouteB_Test/ folder:

    streamlit run step2a_api_test.py

Requires: streamlit, anthropic. API key from the ANTHROPIC_API_KEY
environment variable or Streamlit secrets, exactly as simpleapp714.py.
"""

import csv
import os
import re
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# ---------------------------------------------------------------- constants

HERE = Path(__file__).resolve().parent
BUNDLE_DIR = HERE / "Step2A_RouteB_Test"
KEY_FILE = BUNDLE_DIR / "KEY_do_not_open_until_done.csv"
OUT_DIR = HERE / "Step2A_RouteA_API_Test"

MODEL = "claude-opus-4-8"
MAX_TOKENS = 32000
PRICE_IN = 5.00 / 1_000_000     # dollars per input token, Opus 4.8 standard
PRICE_OUT = 25.00 / 1_000_000   # dollars per output token

# The by-hand browser picks from the 23 July run (ledger Entry 55), for the
# side-by-side agreement column. Draft numbers as they appeared in the bundle.
BROWSER_PICKS = {
    "keystone_ch3": 3,
    "keystone_ch11_first": 3,
    "keystone_ch11b": 3,
    "western_romance_ch2": 3,
    "western_romance_ch10": 3,
    "keystone_ch10b_TUNING_CASE": 3,
    "keystone_ch12_prior_pick": 2,
    "keystone_ch25_prior_pick": 2,
}

SELECTED_RE = re.compile(r"SELECTED:\s*DRAFT\s*(\d+)", re.I)

# ----------------------------------------------------------------- api plumbing
# (mirrors simpleapp714.py exactly)

def clean_api_key(value):
    return (value or "").strip().strip('"').strip("'")

def load_api_key():
    try:
        if "ANTHROPIC_API_KEY" in st.secrets:
            key = clean_api_key(str(st.secrets["ANTHROPIC_API_KEY"]))
            if key:
                return key, "secrets"
    except Exception:
        pass
    key = clean_api_key(os.environ.get("ANTHROPIC_API_KEY", ""))
    if key:
        return key, "environment"
    return "", "none"

def call_model(client, prompt_text, status_slot, adaptive_thinking=True):
    """One streamed call, shaped like simpleapp714.py: model, max_tokens, one
    user message, no system prompt, default temperature. With adaptive
    thinking on it adds thinking {type: adaptive} and, via extra_body,
    output_config {effort: high} -- the only thinking form Opus 4.8 accepts
    and the mirror of the web interface recipe. Streaming is required."""
    status_slot.write("waiting for the model...")
    pieces = []
    kwargs = dict(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt_text}],
    )
    if adaptive_thinking:
        kwargs["thinking"] = {"type": "adaptive"}
        kwargs["extra_body"] = {"output_config": {"effort": "high"}}
    with client.messages.stream(**kwargs) as stream:
        for chunk in stream.text_stream:
            pieces.append(chunk)
            if len(pieces) % 50 == 0:
                status_slot.write(
                    f"writing... {len(''.join(pieces).split())} words so far")
        final = stream.get_final_message()
    text = "\n".join(b.text for b in final.content
                     if getattr(b, "text", None))
    usage = final.usage
    return (text,
            getattr(usage, "input_tokens", 0),
            getattr(usage, "output_tokens", 0))

# ----------------------------------------------------------------- grading

def read_key():
    """Return {chapter: {"independent": bool,
                         "drafts": {n: score}, "top": n}}."""
    if not KEY_FILE.exists():
        return {}
    out = {}
    with open(KEY_FILE, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            chap = row["chapter"]
            n = int(row["bundle_draft"].split()[-1])
            score = int(row["true_score"])
            rec = out.setdefault(chap, {
                "independent": row["independent"].strip().lower() == "yes",
                "drafts": {}})
            rec["drafts"][n] = score
    for rec in out.values():
        rec["top"] = max(rec["drafts"], key=lambda n: rec["drafts"][n])
    return out

def chapter_from_bundle(name):
    """bundle_06_keystone_ch10b_TUNING_CASE.txt -> keystone_ch10b_TUNING_CASE"""
    stem = Path(name).stem
    return re.sub(r"^bundle_\d+_", "", stem)

def parse_selected(text):
    hits = SELECTED_RE.findall(text or "")
    return int(hits[-1]) if hits else None

def grade(pick, drafts):
    """Score a pick against Originality only. Returns a short label. NOT a
    quality judgment -- see the module docstring and ledger Entry 55."""
    if pick is None or pick not in drafts:
        return "no SELECTED line"
    scores = drafts
    top = max(scores.values())
    worst = min(scores.values())
    ps = scores[pick]
    if ps == top:
        return "top score"
    if ps >= top - 2:
        return "1-2 below top (noise)"
    if ps == worst:
        return "lowest score"
    return "middle"

# ----------------------------------------------------------------- the app

def main():
    st.set_page_config(page_title="Step 2A -- Route A (API)", layout="wide")
    st.title("Step 2A -- Route A: API replication")
    st.caption("Does an API assessor pick the same drafts the browser did? "
               "Off-to-the-side experiment. Originality score is not a "
               "quality measure (ledger Entry 55).")

    if not ANTHROPIC_AVAILABLE:
        st.error("The anthropic library is not installed. "
                 "Run: pip install anthropic")
        st.stop()
    if not BUNDLE_DIR.exists():
        st.error(f"No bundle folder at {BUNDLE_DIR}. Run "
                 "step2a_routeb_assemble.py first.")
        st.stop()

    keymap = read_key()
    bundles = sorted(BUNDLE_DIR.glob("bundle_*.txt"))
    if not bundles:
        st.error(f"No bundle_*.txt files in {BUNDLE_DIR}.")
        st.stop()

    api_key, source = load_api_key()
    with st.sidebar:
        st.subheader("Settings")
        if api_key:
            st.success(f"API key loaded from {source}.")
        else:
            api_key = clean_api_key(st.text_input("Anthropic API key",
                                                  type="password"))
        st.write(f"Model: `{MODEL}`, streamed, no system prompt.")
        think = st.checkbox(
            "Adaptive thinking at high effort (mirrors the web interface; "
            "ledger Entry 22). Off = the bare call shape.", value=True)
        runs = st.number_input("Runs per bundle (repeat to see how steady "
                               "the pick is)", 1, 5, 1, 1)

    labels = [b.name for b in bundles]
    chosen = st.multiselect("Bundles to run:", labels, default=labels)

    if st.button("Run the API assessments", type="primary"):
        if not api_key:
            st.error("No API key.")
            st.stop()
        if not chosen:
            st.error("No bundles selected.")
            st.stop()

        client = anthropic.Anthropic(api_key=api_key)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rows = []
        tok_in = tok_out = 0
        t0 = time.time()

        for name in chosen:
            path = BUNDLE_DIR / name
            chap = chapter_from_bundle(name)
            drafts = keymap.get(chap, {}).get("drafts", {})
            independent = keymap.get(chap, {}).get("independent", False)
            bundle_text = path.read_text(encoding="utf-8")
            for r in range(1, int(runs) + 1):
                tag = f"{chap} (run {r}/{int(runs)})"
                with st.status(f"Assessing {tag} ...", expanded=False) as s:
                    text, ti, to = call_model(client, bundle_text, s, think)
                    tok_in += ti
                    tok_out += to
                    pick = parse_selected(text)
                    (OUT_DIR / f"{chap}_run{r}_{stamp}.md").write_text(
                        text, encoding="utf-8")
                    s.update(label=f"{tag} -- API chose DRAFT {pick}",
                             state="complete")
                browser = BROWSER_PICKS.get(chap)
                sc = drafts.get(pick) if pick in drafts else None
                rows.append({
                    "chapter": chap,
                    "independent": "yes" if independent else "no",
                    "run": r,
                    "scores(D1/D2/D3)": "/".join(
                        str(drafts.get(n, "?")) for n in (1, 2, 3)),
                    "browser_pick": f"DRAFT {browser}" if browser else "?",
                    "api_pick": f"DRAFT {pick}" if pick else "none",
                    "api_pick_score": sc if sc is not None else "?",
                    "vs_originality": grade(pick, drafts) if drafts else "?",
                    "agrees_with_browser":
                        "same" if (browser and pick == browser)
                        else ("diff" if pick else "?"),
                })

        cost = tok_in * PRICE_IN + tok_out * PRICE_OUT
        st.session_state["routeA"] = {
            "rows": rows, "cost": round(cost, 2),
            "minutes": round((time.time() - t0) / 60, 1),
            "stamp": stamp,
        }

        # write the results CSV
        csv_path = OUT_DIR / f"routeA_results_{stamp}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        st.session_state["routeA"]["csv"] = str(csv_path)

    # ---- results (rendered from session state so it survives reruns)
    if "routeA" in st.session_state:
        d = st.session_state["routeA"]
        rows = d["rows"]
        st.subheader("Results")
        st.caption("'vs_originality' compares the pick to the highest "
                   "Originality score only. It is NOT a quality judgment "
                   "(ledger Entry 55). 'agrees_with_browser' is the real "
                   "question this app answers.")
        st.dataframe(rows, use_container_width=True)

        same = sum(1 for r in rows if r["agrees_with_browser"] == "same")
        graded = sum(1 for r in rows if r["agrees_with_browser"] in ("same", "diff"))
        ind_top = sum(1 for r in rows
                      if r["independent"] == "yes"
                      and r["vs_originality"] == "top score")
        ind_n = sum(1 for r in rows if r["independent"] == "yes")
        st.write(f"**Agreement with the browser:** {same} of {graded} runs "
                 f"chose the same draft the browser chose.")
        st.write(f"**Independent chapters, top-Originality picks "
                 f"(provisional, not quality):** {ind_top} of {ind_n}.")
        st.write(f"About ${d['cost']:.2f}, {d['minutes']} minutes. "
                 f"Verdicts and `{Path(d['csv']).name}` saved in "
                 f"`{OUT_DIR}`.")

if __name__ == "__main__":
    main()
