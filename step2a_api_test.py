"""
step2a_api_test.py -- ROUTE A: the API replication of the Step 2A trial
=======================================================================

OFF-TO-THE-SIDE experiment, not part of the writing pipeline. It exists only
to check one thing: does an API assessor choose the same drafts a by-hand
browser window chose? Originality is nowhere in it.

It sends each of the eight blinded Step 2A bundles to Claude through the API
-- same streamed call shape as simpleapp714.py, adaptive thinking at high
effort by default (mirrors the web recipe, ledger Entry 22) -- reads the
"SELECTED: DRAFT n" line, and shows the API's pick beside the browser's pick.

The bundles load two ways: automatically from a Step2A_RouteB_Test/ folder
next to this script (a local run), OR by uploading the eight bundle_*.txt
files in the browser (a Streamlit Cloud run, where there is no folder). The
score key and the browser picks are baked in below, so no KEY.csv or folder
is required -- the bundles are the only input.

IMPORTANT (ledger Entry 55): the Originality score is NOT writing quality. A
lower-scoring draft can be the better prose. "top score" below means only
"chose the highest-Originality draft"; it is not a quality verdict. This app
measures agreement with the browser and the score pattern, nothing more.

Run locally:   streamlit run step2a_api_test.py
On the cloud:  push this file, set ANTHROPIC_API_KEY in Streamlit secrets,
               and upload the eight bundle_*.txt files in the app.

Requires: streamlit, anthropic  (put both in requirements.txt).
"""

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
BUNDLE_DIR = HERE / "Step2A_RouteB_Test"      # used only for a local run

MODEL = "claude-opus-4-8"
MAX_TOKENS = 32000
PRICE_IN = 5.00 / 1_000_000
PRICE_OUT = 25.00 / 1_000_000

# The score of each draft in each bundle (from the assembler's KEY, 23 July).
# Baked in so the app needs no KEY.csv. Draft numbers are the blinded
# positions the assessor sees.
KEY_SCORES = {
    "keystone_ch3":               {1: 90, 2: 92, 3: 72},
    "keystone_ch11_first":        {1: 81, 2: 79, 3: 90},
    "keystone_ch11b":             {1: 85, 2: 95, 3: 94},
    "western_romance_ch2":        {1: 81, 2: 85, 3: 80},
    "western_romance_ch10":       {1: 92, 2: 75, 3: 95},
    "keystone_ch10b_TUNING_CASE": {1: 84, 2: 97, 3: 91},
    "keystone_ch12_prior_pick":   {1: 91, 2: 97, 3: 90},
    "keystone_ch25_prior_pick":   {1: 76, 2: 93, 3: 33},
}
INDEPENDENT = {
    "keystone_ch3", "keystone_ch11_first", "keystone_ch11b",
    "western_romance_ch2", "western_romance_ch10",
}
# The by-hand browser picks from the 23 July run (ledger Entry 55).
BROWSER_PICKS = {
    "keystone_ch3": 3, "keystone_ch11_first": 3, "keystone_ch11b": 3,
    "western_romance_ch2": 3, "western_romance_ch10": 3,
    "keystone_ch10b_TUNING_CASE": 3, "keystone_ch12_prior_pick": 2,
    "keystone_ch25_prior_pick": 2,
}

SELECTED_RE = re.compile(r"SELECTED:\s*DRAFT\s*(\d+)", re.I)
OUT_DIR = HERE / "Step2A_RouteA_API_Test"

# ----------------------------------------------------------------- api plumbing

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
    """One streamed call, shaped like simpleapp714.py."""
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

def chapter_from_bundle(name):
    """bundle_06_keystone_ch10b_TUNING_CASE.txt -> keystone_ch10b_TUNING_CASE"""
    return re.sub(r"^bundle_\d+_", "", Path(name).stem)

def parse_selected(text):
    hits = SELECTED_RE.findall(text or "")
    return int(hits[-1]) if hits else None

def grade(pick, scores):
    """Score a pick against Originality only -- NOT a quality judgment."""
    if pick is None or pick not in scores:
        return "no SELECTED line"
    top, worst, ps = max(scores.values()), min(scores.values()), scores[pick]
    if ps == top:
        return "top score"
    if ps >= top - 2:
        return "1-2 below top (noise)"
    if ps == worst:
        return "lowest score"
    return "middle"

def load_bundles(uploaded):
    """Return [(name, text)] from the uploaded files, or the local folder."""
    if uploaded:
        return [(uf.name, uf.getvalue().decode("utf-8", "replace"))
                for uf in uploaded]
    if BUNDLE_DIR.exists():
        return [(p.name, p.read_text(encoding="utf-8"))
                for p in sorted(BUNDLE_DIR.glob("bundle_*.txt"))]
    return []

# ----------------------------------------------------------------- the app

def main():
    st.set_page_config(page_title="Step 2A -- Route A (API)", layout="wide")
    st.title("Step 2A -- Route A: API replication")
    st.caption("Does an API assessor pick the same drafts the browser did? "
               "Off-to-the-side experiment. Originality score is not a "
               "quality measure (ledger Entry 55).")

    if not ANTHROPIC_AVAILABLE:
        st.error("The anthropic library is not installed. Add 'anthropic' "
                 "to requirements.txt.")
        st.stop()

    st.write("**Step 1 -- give it the bundles.** Upload the eight "
             "`bundle_*.txt` files (or drop a `Step2A_RouteB_Test/` folder "
             "next to this script for a local run).")
    uploaded = st.file_uploader("Bundle .txt files", type=["txt"],
                                accept_multiple_files=True)
    items = load_bundles(uploaded)
    if not items:
        st.info("Upload the eight bundle files to begin.")
        st.stop()
    st.success(f"{len(items)} bundle(s) loaded.")

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
            "ledger Entry 22). If a call errors, turn this OFF.", value=True)
        runs = st.number_input("Runs per bundle", 1, 5, 1, 1)

    names = [n for n, _ in items]
    chosen = st.multiselect("Bundles to run:", names, default=names)

    if st.button("Run the API assessments", type="primary"):
        if not api_key:
            st.error("No API key.")
            st.stop()
        if not chosen:
            st.error("No bundles selected.")
            st.stop()

        text_by_name = dict(items)
        client = anthropic.Anthropic(api_key=api_key)
        try:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            can_save = True
        except Exception:
            can_save = False   # read-only cloud disk; results still render
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rows, tok_in, tok_out, t0 = [], 0, 0, time.time()

        for name in chosen:
            chap = chapter_from_bundle(name)
            scores = KEY_SCORES.get(chap, {})
            bundle_text = text_by_name[name]
            for r in range(1, int(runs) + 1):
                tag = f"{chap} (run {r}/{int(runs)})"
                with st.status(f"Assessing {tag} ...", expanded=False) as s:
                    text, ti, to = call_model(client, bundle_text, s, think)
                    tok_in += ti
                    tok_out += to
                    pick = parse_selected(text)
                    if can_save:
                        try:
                            (OUT_DIR / f"{chap}_run{r}_{stamp}.md").write_text(
                                text, encoding="utf-8")
                        except Exception:
                            pass
                    s.update(label=f"{tag} -- API chose DRAFT {pick}",
                             state="complete")
                browser = BROWSER_PICKS.get(chap)
                sc = scores.get(pick)
                rows.append({
                    "chapter": chap,
                    "independent": "yes" if chap in INDEPENDENT else "no",
                    "run": r,
                    "scores(D1/D2/D3)": "/".join(
                        str(scores.get(n, "?")) for n in (1, 2, 3)),
                    "browser_pick": f"DRAFT {browser}" if browser else "?",
                    "api_pick": f"DRAFT {pick}" if pick else "none",
                    "api_pick_score": sc if sc is not None else "?",
                    "vs_originality": grade(pick, scores) if scores else "?",
                    "agrees_with_browser":
                        "same" if (browser and pick == browser)
                        else ("diff" if pick else "?"),
                })

        cost = tok_in * PRICE_IN + tok_out * PRICE_OUT
        st.session_state["routeA"] = {
            "rows": rows, "cost": round(cost, 2),
            "minutes": round((time.time() - t0) / 60, 1),
        }

    if "routeA" in st.session_state:
        d = st.session_state["routeA"]
        rows = d["rows"]
        st.subheader("Results")
        st.caption("'vs_originality' compares the pick to the highest "
                   "Originality score only -- NOT a quality judgment (ledger "
                   "Entry 55). 'agrees_with_browser' is the real question.")
        st.dataframe(rows, use_container_width=True)

        same = sum(1 for r in rows if r["agrees_with_browser"] == "same")
        graded = sum(1 for r in rows
                     if r["agrees_with_browser"] in ("same", "diff"))
        ind_top = sum(1 for r in rows if r["independent"] == "yes"
                      and r["vs_originality"] == "top score")
        ind_n = sum(1 for r in rows if r["independent"] == "yes")
        st.write(f"**Agreement with the browser:** {same} of {graded} runs "
                 f"chose the same draft.")
        st.write(f"**Independent chapters, top-Originality picks "
                 f"(provisional, not quality):** {ind_top} of {ind_n}.")
        st.write(f"About ${d['cost']:.2f}, {d['minutes']} minutes.")

        import io, csv as _csv
        buf = io.StringIO()
        w = _csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
        st.download_button("Download results CSV", buf.getvalue(),
                           file_name="routeA_results.csv", key="dl_csv")

if __name__ == "__main__":
    main()
