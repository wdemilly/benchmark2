"""
Mechanical-spec scanner for AI-assisted drafts.

Reads a per-author TOML spec and runs countable, regex-style checks against
a draft. Returns per-axis pass/fail results suitable for Streamlit display.

Usage:
    from scanner import load_spec, run_scan, summarize_pass_fail
    spec = load_spec("specs/dare.toml")
    scan = run_scan(draft_text, spec)
    rows = summarize_pass_fail(scan)
"""

import re
from collections import Counter

try:
    import tomllib  # Python 3.11+ stdlib
except ImportError:
    import tomli as tomllib  # pip install tomli for Python 3.10


def load_spec(path):
    with open(path, "rb") as f:
        return tomllib.load(f)


# ---------- low-level helpers ----------

def _normalize_quotes(text):
    return (text
            .replace("\u201c", '"').replace("\u201d", '"')
            .replace("\u2018", "'").replace("\u2019", "'"))


def split_sentences(text):
    """Sentence split on .!? followed by space and uppercase/quote."""
    text = _normalize_quotes(text)
    sents = re.split(r'(?<=[\.\!\?])\s+(?=["\']?[A-Z])', text)
    return [s.strip() for s in sents if s.strip() and len(s.strip()) > 1]


def words(text):
    return re.findall(r"\b[\w']+\b", text)


def has_em_dash(s):
    return "\u2014" in s or " -- " in s


# ---------- axis checks ----------

def sentence_length_stats(text, spec):
    sents = split_sentences(text)
    lens = [len(words(s)) for s in sents]
    if not lens:
        return {"sentence_count": 0}
    mean = sum(lens) / len(lens)
    bins = Counter()
    for L in lens:
        if L <= 3: bins["frag"] += 1
        elif L <= 7: bins["short"] += 1
        elif L <= 14: bins["midshort"] += 1
        elif L <= 24: bins["midlong"] += 1
        elif L <= 40: bins["long"] += 1
        else: bins["vlong"] += 1
    total = len(lens)
    pct = {k: 100 * v / total for k, v in bins.items()}
    sl = spec["sentence_length"]
    return {
        "mean": round(mean, 2),
        "mean_pass": sl["mean_min"] <= mean <= sl["mean_max"],
        "frag_pct": round(pct.get("frag", 0), 1),
        "frag_pass": sl["fragments_pct_min"] <= pct.get("frag", 0) <= sl["fragments_pct_max"],
        "vlong_pct": round(pct.get("vlong", 0), 1),
        "vlong_pass": pct.get("vlong", 0) <= sl["very_long_pct_max"],
        "distribution": {k: round(v, 1) for k, v in pct.items()},
        "sentence_count": total,
    }


def em_dash_stats(text, spec):
    em_count = text.count("\u2014") + text.count(" -- ")
    word_count = len(words(text))
    per_1000 = em_count / word_count * 1000 if word_count else 0
    ed = spec["em_dash"]
    target_lo, target_hi = ed["target_per_1000"]
    cap = ed["hard_cap_per_1000"]

    sents = split_sentences(text)
    consec = 0
    max_consec = 0
    for s in sents:
        if has_em_dash(s):
            consec += 1
            max_consec = max(max_consec, consec)
        else:
            consec = 0

    return {
        "count": em_count,
        "per_1000": round(per_1000, 1),
        "in_target": target_lo <= per_1000 <= target_hi,
        "under_cap": per_1000 <= cap,
        "max_consecutive_em_sents": max_consec,
        "no_stacking_pass": max_consec <= 1,
        "mode": ed["mode"],
    }


def banned_pattern_check(text, spec):
    text_norm = _normalize_quotes(text)
    patterns = {
        "the_way": (r"\bthe way\b", re.IGNORECASE),
        "not_but": (r"\bnot\b[^.!?\n]{1,60}\bbut\b", re.IGNORECASE),
        "not_x_not_y": (r"\bNot\s+\w+[^.!?\n]{0,40}\.\s*Not\s+\w", 0),
    }
    bp = spec["banned_patterns"]
    out = {}
    for name, (pat, flags) in patterns.items():
        matches = re.findall(pat, text_norm, flags)
        allowed = bp.get(name, 0)
        out[name] = {
            "count": len(matches),
            "allowed": allowed,
            "pass": len(matches) <= allowed,
            "examples": matches[:3] if matches else [],
        }
    return out


def fragment_clusters(text, spec):
    sents = split_sentences(text)
    lens = [len(words(s)) for s in sents]
    clusters = []
    i = 0
    while i < len(lens):
        if lens[i] <= 3:
            j = i
            while j + 1 < len(lens) and lens[j + 1] <= 3:
                j += 1
            if j > i:
                clusters.append(j - i + 1)
            i = j + 1
        else:
            i += 1
    fc = spec["fragment_clusters"]
    max_observed = max(clusters) if clusters else 0
    target_lo, target_hi = fc["target_count_per_4000_words"]
    word_count = len(words(text))
    expected_target = (target_lo * word_count / 4000, target_hi * word_count / 4000)
    return {
        "cluster_count": len(clusters),
        "cluster_lengths": clusters,
        "max_length": max_observed,
        "max_length_pass": max_observed <= fc["max_run_length"],
        "count_in_target": expected_target[0] <= len(clusters) <= expected_target[1] + 1,
        "scaled_target": (round(expected_target[0], 1), round(expected_target[1], 1)),
    }


def anaphora_runs(text, spec):
    stop = {"the", "a", "an", "and", "but", "it", "he", "she", "i", "they",
            "his", "her", "this", "that", "was", "were", "to", "of", "in", "on", "at"}
    sents = split_sentences(text)
    starts = []
    for s in sents:
        w = s.split()
        if w:
            starts.append(re.sub(r"[^\w]", "", w[0]).lower())
        else:
            starts.append("")
    runs = []
    i = 0
    while i < len(starts):
        j = i
        while (j + 1 < len(starts)
               and starts[j + 1] == starts[i]
               and starts[i]
               and starts[i] not in stop):
            j += 1
        if j > i:
            runs.append((starts[i], j - i + 1))
        i = j + 1
    a = spec["anaphora"]
    longest = max((n for _, n in runs), default=0)
    return {
        "runs": runs,
        "count": len(runs),
        "longest": longest,
        "longest_pass": longest <= a["max_run"],
        "count_pass": len(runs) <= a["max_runs_per_chapter"],
    }


def paragraph_stats(text, spec):
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    plens = [len(words(p)) for p in paras]
    if not plens:
        return {"count": 0}
    p = spec["paragraphs"]
    short = sum(1 for L in plens if L < 50)
    long_ = sum(1 for L in plens if L > 250)
    var_violations = 0
    for i in range(len(plens) - 1):
        a, b = plens[i], plens[i + 1]
        if a == 0 or b == 0:
            continue
        smaller, larger = min(a, b), max(a, b)
        if larger - smaller < smaller * (p["consecutive_variance_min_pct"] / 100):
            var_violations += 1
    consec_default = 0
    max_consec_default = 0
    for L in plens:
        if 100 <= L <= 250:
            consec_default += 1
            max_consec_default = max(max_consec_default, consec_default)
        else:
            consec_default = 0
    return {
        "count": len(paras),
        "mean_words": round(sum(plens) / len(plens), 1),
        "short_count": short,
        "short_pass": short >= p["short_under_50_min_count"],
        "long_count": long_,
        "long_pass": long_ >= p["long_over_250_min_count"],
        "consecutive_sameness_violations": var_violations,
        "max_consec_default_band": max_consec_default,
        "default_band_pass": max_consec_default <= p["max_consecutive_in_default_band"],
        "lengths": plens,
    }


def punctuation_caps(text, spec):
    semis = text.count(";")
    colons = text.count(":") - len(re.findall(r"said\s*:", text, re.IGNORECASE))
    adv_tags = len(re.findall(r"\bsaid\s+\w+ly\b", text, re.IGNORECASE))
    parens = text.count("(")
    c = spec["caps"]
    return {
        "semicolons": semis, "semicolons_pass": semis <= c["semicolons"],
        "colons": max(colons, 0), "colons_pass": max(colons, 0) <= c["colons"],
        "adverb_tags": adv_tags, "adverb_tags_pass": adv_tags <= c["adverb_tags"],
        "parens": parens, "parens_pass": parens <= c["parentheticals"],
    }


# ---------- top-level ----------

def run_scan(text, spec):
    return {
        "sentence_length": sentence_length_stats(text, spec),
        "em_dash": em_dash_stats(text, spec),
        "banned": banned_pattern_check(text, spec),
        "fragments": fragment_clusters(text, spec),
        "anaphora": anaphora_runs(text, spec),
        "paragraphs": paragraph_stats(text, spec),
        "punctuation": punctuation_caps(text, spec),
    }


def summarize_pass_fail(scan):
    """Flat list of (axis, value, status) tuples for table display."""
    rows = []
    sl = scan["sentence_length"]
    if sl.get("sentence_count"):
        rows.append(("Mean sentence length", f"{sl['mean']} words",
                     "PASS" if sl["mean_pass"] else "FAIL"))
        rows.append(("Fragment %", f"{sl['frag_pct']}%",
                     "PASS" if sl["frag_pass"] else "FAIL"))
        rows.append(("Very-long sentence %", f"{sl['vlong_pct']}%",
                     "PASS" if sl["vlong_pass"] else "FAIL"))
    ed = scan["em_dash"]
    rows.append((f"Em-dashes / 1000 words ({ed['mode']})",
                 f"{ed['per_1000']} ({ed['count']} total)",
                 "PASS" if ed["under_cap"] else "FAIL"))
    rows.append(("Em-dash sentence stacking",
                 f"max {ed['max_consecutive_em_sents']} consecutive",
                 "PASS" if ed["no_stacking_pass"] else "FAIL"))
    fc = scan["fragments"]
    rows.append(("Fragment cluster max length", str(fc["max_length"]),
                 "PASS" if fc["max_length_pass"] else "FAIL"))
    rows.append(("Fragment cluster count",
                 f"{fc['cluster_count']} (target {fc['scaled_target']})",
                 "PASS" if fc["count_in_target"] else "INFO"))
    an = scan["anaphora"]
    rows.append(("Anaphora longest run", str(an["longest"]),
                 "PASS" if an["longest_pass"] else "FAIL"))
    rows.append(("Anaphora run count", str(an["count"]),
                 "PASS" if an["count_pass"] else "FAIL"))
    for name, r in scan["banned"].items():
        rows.append((f"Banned: {name}", str(r["count"]),
                     "PASS" if r["pass"] else "FAIL"))
    pa = scan["paragraphs"]
    if pa.get("count"):
        rows.append(("Short paragraphs (<50 wd)", str(pa["short_count"]),
                     "PASS" if pa["short_pass"] else "FAIL"))
        rows.append(("Long paragraphs (>250 wd)", str(pa["long_count"]),
                     "PASS" if pa["long_pass"] else "FAIL"))
        rows.append(("Consecutive-para sameness",
                     f"{pa['consecutive_sameness_violations']} pairs", "INFO"))
        rows.append(("Max consec paras in default band",
                     str(pa["max_consec_default_band"]),
                     "PASS" if pa["default_band_pass"] else "FAIL"))
    pn = scan["punctuation"]
    rows.append(("Semicolons", str(pn["semicolons"]),
                 "PASS" if pn["semicolons_pass"] else "FAIL"))
    rows.append(("Colons", str(pn["colons"]),
                 "PASS" if pn["colons_pass"] else "FAIL"))
    rows.append(("Adverb-loaded dialogue tags", str(pn["adverb_tags"]),
                 "PASS" if pn["adverb_tags_pass"] else "FAIL"))
    rows.append(("Parentheticals", str(pn["parens"]),
                 "PASS" if pn["parens_pass"] else "FAIL"))
    return rows


def fail_count(rows):
    return sum(1 for _, _, s in rows if s == "FAIL")
