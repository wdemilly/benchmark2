"""
predict_single.py

Predict the Originality human-score for a single draft file.

Usage:
    python predict_single.py --draft path/to/draft.txt
    python predict_single.py --draft path/to/FINAL.txt --model claude-sonnet-4-6 --k 5

Accepts .txt, .md, or .docx. Prints the predicted score, the ridge and
NN-mean baselines, the neighbors used, and the rationale. Exits non-zero
if the LLM fails to return a parseable score.
"""

from __future__ import annotations
import argparse
import io
import json
import os
import sys
import zipfile
import re
from pathlib import Path

from originality_predictor import OriginalityPredictor


def read_draft(path: Path) -> str:
    if path.suffix.lower() == ".docx":
        with open(path, "rb") as f:
            data = f.read()
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            with z.open("word/document.xml") as fh:
                xml = fh.read().decode("utf-8", errors="replace")
        text = re.sub(r"<[^>]+>", " ", xml)
        text = re.sub(r"\s+", " ", text).strip()
        import html
        return html.unescape(text)
    return path.read_text(encoding="utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", required=True, type=Path, help="Path to draft (.txt/.md/.docx)")
    ap.add_argument("--corpus", default="labeled_corpus.json", type=Path)
    ap.add_argument("--model", default="claude-opus-4-7")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--json", action="store_true", help="Emit the full result as JSON")
    args = ap.parse_args()

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: need ANTHROPIC_API_KEY env var or --api-key", file=sys.stderr)
        sys.exit(2)

    try:
        import anthropic  # type: ignore
    except ImportError:
        print("ERROR: pip install anthropic", file=sys.stderr)
        sys.exit(2)

    client = anthropic.Anthropic(api_key=api_key)
    predictor = OriginalityPredictor(args.corpus)

    text = read_draft(args.draft)
    wc = len(text.split())
    result = predictor.predict(
        draft_text=text,
        client=client,
        model=args.model,
        k=args.k,
    )

    if args.json:
        # Strip large text fields before dumping
        out = dict(result)
        out["neighbors"] = [
            {kk: vv for kk, vv in nb.items() if kk != "text"}
            for nb in out["neighbors"]
        ]
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return

    pred = result["predicted_score"]
    print(f"Draft:    {args.draft.name}  ({wc} words)")
    print(f"Model:    {args.model}  |  k={args.k}")
    print(f"Tokens:   ~{result['input_token_estimate']} input")
    print()
    print(f"Predicted Originality human-score:  {pred}")
    print(f"Recommendation:                     {predictor.recommendation(pred)}")
    print(f"Ridge baseline:                     {result['style_baseline_score']}")
    print(f"NN-mean baseline:                   {result['nn_mean_baseline']}")
    print()
    print("Neighbors used:")
    for nb in result["neighbors"]:
        print(f"  sim={nb['similarity']:.3f}  score={nb['human_score']:3d}  "
              f"wc={nb['word_count']:>5}  {nb['source_file']}")
    print()
    print("Rationale:")
    print(result["rationale"])

    if result["parse_status"] != "ok":
        print(f"\nWARN: parse_status={result['parse_status']}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
