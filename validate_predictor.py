"""
validate_predictor.py

Run the LLM predictor in leave-one-out mode against a sample of the
labeled corpus. Reports MAE, Pearson r, and gate precision against the
held-out human-scores. Compares LLM predictions against the closed-form
ridge and NN-mean baselines computed inside the predictor.

Cost note: each prediction call is ~$0.35 on Opus, ~$0.07 on Sonnet.
The default --n 10 keeps cost under $4 on Opus, under $1 on Sonnet.
Use --n 48 for a full leave-one-out (the whole corpus), ~$17 on Opus.

Requires: ANTHROPIC_API_KEY in env, or pass --api-key on the command line.

Usage:
    python validate_predictor.py --corpus labeled_corpus.json --n 10
    python validate_predictor.py --corpus labeled_corpus.json --n 20 --model claude-sonnet-4-6
    python validate_predictor.py --corpus labeled_corpus.json --all --model claude-sonnet-4-6
"""

from __future__ import annotations
import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

from originality_predictor import OriginalityPredictor


def stratified_sample(records: list[dict], n: int, seed: int = 42) -> list[int]:
    """Return indices of a stratified-by-score sample of size n."""
    rng = random.Random(seed)
    bands = {"0-20": [], "21-40": [], "41-60": [], "61-80": [], "81-100": []}
    for i, r in enumerate(records):
        s = r["human_score"]
        if s <= 20:
            bands["0-20"].append(i)
        elif s <= 40:
            bands["21-40"].append(i)
        elif s <= 60:
            bands["41-60"].append(i)
        elif s <= 80:
            bands["61-80"].append(i)
        else:
            bands["81-100"].append(i)

    per_band = max(1, n // 5)
    picked = []
    for ids in bands.values():
        rng.shuffle(ids)
        picked.extend(ids[:per_band])

    # Top up to n if the band splitting left us short
    remaining = [i for i in range(len(records)) if i not in picked]
    rng.shuffle(remaining)
    while len(picked) < n and remaining:
        picked.append(remaining.pop(0))

    return picked[:n]


def metrics(preds: list[float], targets: list[int]) -> dict:
    preds_np = np.array(preds, dtype=float)
    targets_np = np.array(targets, dtype=float)
    mask = ~np.isnan(preds_np)
    if mask.sum() == 0:
        return {"n": 0}

    p = preds_np[mask]
    t = targets_np[mask]
    mae = float(np.mean(np.abs(p - t)))
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    r = float(np.corrcoef(p, t)[0, 1]) if np.std(p) > 0 and np.std(t) > 0 else 0.0

    out = {"n": int(mask.sum()), "mae": round(mae, 2), "rmse": round(rmse, 2), "r": round(r, 3)}

    # Gate precision
    for gate in [80, 85, 88, 90, 92]:
        ships = p >= gate
        if ships.sum() == 0:
            out[f"gate_{gate}"] = None
            continue
        hit85 = int(((p >= gate) & (t >= 85)).sum())
        hit90 = int(((p >= gate) & (t >= 90)).sum())
        n = int(ships.sum())
        out[f"gate_{gate}"] = {
            "ships": n,
            "real_ge_85_pct": round(100 * hit85 / n, 1),
            "real_ge_90_pct": round(100 * hit90 / n, 1),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="labeled_corpus.json", type=Path)
    ap.add_argument("--n", type=int, default=10,
                    help="Number of held-out drafts to test (default 10, stratified).")
    ap.add_argument("--all", action="store_true",
                    help="Run on the full corpus (LOO). Overrides --n.")
    ap.add_argument("--k", type=int, default=5, help="Neighbors per prediction (default 5)")
    ap.add_argument("--model", default="claude-opus-4-7")
    ap.add_argument("--api-key", default=None,
                    help="Anthropic API key (defaults to ANTHROPIC_API_KEY env).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("validation_results.csv"))
    ap.add_argument("--sleep", type=float, default=0.0,
                    help="Sleep seconds between API calls (rate limiting).")
    args = ap.parse_args()

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: need ANTHROPIC_API_KEY env var or --api-key", file=sys.stderr)
        sys.exit(1)

    try:
        import anthropic  # type: ignore
    except ImportError:
        print("ERROR: pip install anthropic", file=sys.stderr)
        sys.exit(1)
    client = anthropic.Anthropic(api_key=api_key)

    predictor = OriginalityPredictor(args.corpus)
    records = predictor.records
    print(f"Loaded {len(records)} labeled records from {args.corpus}")

    if args.all:
        indices = list(range(len(records)))
    else:
        indices = stratified_sample(records, args.n, seed=args.seed)

    print(f"Testing on {len(indices)} drafts. Model={args.model}. k={args.k}")
    print(f"Est. cost: ~${0.35 * len(indices):.2f} (Opus) / ~${0.07 * len(indices):.2f} (Sonnet)")
    print()

    rows = []
    llm_preds = []
    ridge_preds = []
    nn_preds = []
    targets = []

    for i, idx in enumerate(indices, 1):
        rec = records[idx]
        print(f"[{i}/{len(indices)}] id={rec['id']} actual={rec['human_score']} ... ", end="", flush=True)
        t0 = time.time()
        pred = None
        ridge = None
        nn = None
        parse_status = "api_error"
        try:
            result = predictor.predict(
                draft_text=rec["text"],
                client=client,
                model=args.model,
                k=args.k,
                exclude_ids={rec["id"]},
            )
            pred = result["predicted_score"]
            ridge = result["style_baseline_score"]
            nn = result["nn_mean_baseline"]
            parse_status = result.get("parse_status", "unknown")
            dt = time.time() - t0
            print(f"llm={pred}  ridge={ridge:.0f}  nn={nn:.0f}  ({dt:.1f}s)")
        except Exception as e:
            print(f"ERROR: {e}")

        rows.append({
            "id": rec["id"],
            "actual": rec["human_score"],
            "llm_pred": pred if pred is not None else "",
            "ridge_pred": ridge if ridge is not None else "",
            "nn_mean_pred": nn if nn is not None else "",
            "abs_error_llm": abs(pred - rec["human_score"]) if pred is not None else "",
            "parse_status": parse_status,
        })
        targets.append(rec["human_score"])
        llm_preds.append(pred if pred is not None else np.nan)
        ridge_preds.append(ridge if ridge is not None else np.nan)
        nn_preds.append(nn if nn is not None else np.nan)

        if args.sleep > 0:
            time.sleep(args.sleep)

    # Write CSV
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nResults -> {args.out}")

    # Metrics
    print("\n" + "=" * 70)
    print("METRICS")
    print("=" * 70)
    print(f"\nLLM predictor ({args.model}, k={args.k}):")
    print(json.dumps(metrics(llm_preds, targets), indent=2))
    print(f"\nRidge baseline (closed-form, no LLM):")
    print(json.dumps(metrics(ridge_preds, targets), indent=2))
    print(f"\nNN-mean baseline (neighbor score mean, no LLM):")
    print(json.dumps(metrics(nn_preds, targets), indent=2))


if __name__ == "__main__":
    main()
