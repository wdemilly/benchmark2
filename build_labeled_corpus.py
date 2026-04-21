"""
build_labeled_corpus.py

One-time preprocessor. Walks the provided Originality-export docx folders,
parses headline human-scores from filenames, extracts plain text and
color metrics, and writes labeled_corpus.json.

Filename convention (observed):
  - "export NN.docx"         -> human_score = NN
  - "export NNAI.docx"       -> ai_score = NN, so human_score = 100 - NN
  - "export(XX) NN.docx"     -> paren is a run/export ID; NN is the score
  - "NN N NN.docx" etc       -> multiple numbers; score is the last one 0-100
  - "export top 1..5.docx"   -> "top N" is a rank, NOT a score -> UNLABELED
  - "export(N).docx" etc     -> UNLABELED (no score in filename)

Usage:
    python build_labeled_corpus.py \
        --in /path/to/validation_exports \
        --in /path/to/more_exports \
        --out labeled_corpus.json

The JSON is a list of records:
    {
      "id": "val2__export_87",
      "source_file": "export 87.docx",
      "source_folder": "validation_2",
      "human_score": 87,
      "label_type": "score_in_name",
      "text": "Chapter One... (full draft text)",
      "word_count": 2715,
      "color_metrics": {
          "segs": 231, "strong_green": 107, "mild_green": 37,
          "neutral": 28, "mild_orange": 27, "strong_orange": 32,
          "green_pct": 62.3, "orange_pct": 25.5
      }
    }
"""

from __future__ import annotations
import argparse
import html
import io
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Docx parsing (borrowed from simpleapp_v15.py; kept standalone for portability)
# ---------------------------------------------------------------------------

_ORIG_HEX_FILL_RE = re.compile(r'w:fill="([0-9A-Fa-f]{6})"')


def _classify_fill(hex_color: str) -> str:
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    diff = g - r
    if diff >= 15:
        return "STRONG_GREEN"
    if diff >= 5:
        return "mild_green"
    if diff >= -5:
        return "neutral"
    if diff >= -15:
        return "mild_orange"
    return "STRONG_ORANGE"


def _read_docx_xml(path: Path) -> str:
    with open(path, "rb") as f:
        data = f.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        with z.open("word/document.xml") as fh:
            return fh.read().decode("utf-8", errors="replace")


def _extract_text_from_xml(xml: str) -> str:
    text = re.sub(r"<[^>]+>", " ", xml)
    text = re.sub(r"\s+", " ", text).strip()
    return html.unescape(text)


def _extract_fills(xml: str) -> list[str]:
    return _ORIG_HEX_FILL_RE.findall(xml)


def _color_metrics(fills: list[str]) -> dict:
    classes = [_classify_fill(h) for h in fills]
    counts = Counter(classes)
    total = len(classes) or 1
    sg = counts.get("STRONG_GREEN", 0)
    mg = counts.get("mild_green", 0)
    n = counts.get("neutral", 0)
    mo = counts.get("mild_orange", 0)
    so = counts.get("STRONG_ORANGE", 0)
    return {
        "segs": len(classes),
        "strong_green": sg,
        "mild_green": mg,
        "neutral": n,
        "mild_orange": mo,
        "strong_orange": so,
        "green_pct": round(100 * (sg + mg) / total, 1),
        "orange_pct": round(100 * (so + mo) / total, 1),
    }


# ---------------------------------------------------------------------------
# Filename -> score parsing
# ---------------------------------------------------------------------------

def parse_score_from_filename(name: str) -> tuple[int | None, str]:
    """Return (human_score, label_type). label_type is one of:
        score_in_name, ai_in_name, top_rank, unlabeled.
    """
    stem = name.replace(".docx", "")

    # TOP ranking files: "export top 1.docx" — N is rank, not score
    if re.search(r"\btop\s+\d+\b", stem, re.IGNORECASE):
        return (None, "top_rank")

    # AI score in name: "NNAI" or "NN AI"
    ai_match = re.search(r"(\d{1,3})\s*AI\b", stem, re.IGNORECASE)
    if ai_match:
        ai = int(ai_match.group(1))
        if 0 <= ai <= 100:
            return (100 - ai, "ai_in_name")

    # Strip paren contents (they hold file/run IDs like "(51)", "(90)")
    cleaned = re.sub(r"\([^)]*\)", "", stem)
    # Strip replication tags like R1, R2
    cleaned = re.sub(r"\bR\d\b", "", cleaned, flags=re.IGNORECASE)

    # Collect any 1-3 digit numbers in the cleaned stem
    nums = re.findall(r"\d{1,3}", cleaned)
    valid = [int(x) for x in nums if 0 <= int(x) <= 100]

    if valid:
        # The *last* number in the cleaned name is overwhelmingly the score
        # (validated empirically: Pearson r score vs green%-orange% = +0.826)
        return (valid[-1], "score_in_name")

    # Fallback: trailing digits stuck to text like "export83"
    trail = re.search(r"(\d{1,3})$", cleaned.strip())
    if trail:
        v = int(trail.group(1))
        if 0 <= v <= 100:
            return (v, "score_in_name")

    return (None, "unlabeled")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_corpus(
    input_dirs: list[tuple[str, Path]],
    out_path: Path,
    include_unlabeled: bool = False,
) -> dict:
    """Scan input_dirs (list of (tag, Path)) and write labeled_corpus.json.
    Returns a summary dict.
    """
    records = []
    summary = {
        "total_files": 0,
        "labeled": 0,
        "unlabeled": 0,
        "top_rank_skipped": 0,
        "by_source": {},
        "score_distribution": {},
    }

    for tag, dirpath in input_dirs:
        if not dirpath.exists():
            print(f"[warn] missing folder: {dirpath}")
            continue
        files_in_dir = sorted(dirpath.glob("*.docx"))
        summary["by_source"][tag] = {"total": len(files_in_dir), "labeled": 0}

        for f in files_in_dir:
            summary["total_files"] += 1
            score, label_type = parse_score_from_filename(f.name)

            if label_type == "top_rank":
                summary["top_rank_skipped"] += 1
                continue

            if score is None and not include_unlabeled:
                summary["unlabeled"] += 1
                continue

            try:
                xml = _read_docx_xml(f)
                text = _extract_text_from_xml(xml)
                fills = _extract_fills(xml)
                metrics = _color_metrics(fills)
            except Exception as e:
                print(f"[error] {f.name}: {e}")
                continue

            rec_id = f"{tag}__{re.sub(r'[^A-Za-z0-9]+', '_', f.stem).strip('_')}"
            records.append({
                "id": rec_id,
                "source_file": f.name,
                "source_folder": tag,
                "human_score": score,
                "label_type": label_type,
                "text": text,
                "word_count": len(text.split()),
                "color_metrics": metrics,
            })
            if score is not None:
                summary["labeled"] += 1
                summary["by_source"][tag]["labeled"] += 1

    # Deduplicate: only collapse records whose FULL normalized text is identical.
    # We deliberately do NOT collapse graft pairs (which share 90%+ of their
    # text but differ in a handful of sentences) — each has distinct per-
    # sentence color data that matters for sentence-level training.
    seen = {}
    unique_records = []
    dupes_removed = 0
    for r in records:
        fp = re.sub(r"\s+", " ", r["text"]).strip().lower()
        if fp in seen:
            dupes_removed += 1
            continue
        seen[fp] = r["id"]
        unique_records.append(r)
    summary["duplicates_removed"] = dupes_removed
    summary["unique_records"] = len(unique_records)

    # Score histogram
    scores = [r["human_score"] for r in unique_records if r["human_score"] is not None]
    if scores:
        bins = {"0-20": 0, "21-40": 0, "41-60": 0, "61-80": 0, "81-100": 0}
        for s in scores:
            if s <= 20:
                bins["0-20"] += 1
            elif s <= 40:
                bins["21-40"] += 1
            elif s <= 60:
                bins["41-60"] += 1
            elif s <= 80:
                bins["61-80"] += 1
            else:
                bins["81-100"] += 1
        summary["score_distribution"] = bins
        summary["score_min"] = min(scores)
        summary["score_max"] = max(scores)
        summary["score_mean"] = round(sum(scores) / len(scores), 1)

    out_path.write_text(json.dumps(unique_records, indent=2, ensure_ascii=False))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inputs", action="append", required=True,
                    help="tag=path, e.g. validation_1=/path/to/folder. Repeat for multiple.")
    ap.add_argument("--out", type=Path, default=Path("labeled_corpus.json"))
    ap.add_argument("--include-unlabeled", action="store_true",
                    help="Include records with no parseable score. They are stored "
                         "with human_score=null and are ignored by the predictor.")
    args = ap.parse_args()

    input_dirs = []
    for entry in args.inputs:
        if "=" in entry:
            tag, path = entry.split("=", 1)
        else:
            tag = Path(entry).name
            path = entry
        input_dirs.append((tag, Path(path)))

    summary = build_corpus(input_dirs, args.out, include_unlabeled=args.include_unlabeled)
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
