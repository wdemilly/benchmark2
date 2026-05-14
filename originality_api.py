"""
originality_api.py — Originality.ai API wrapper for Phase 2
============================================================

Single public function: score_text(text) -> (score, response_dict).

Returns the turbo human-probability score (0-100) and the full
response dict so callers can persist the color-banded metadata
the operator uses for diagnostic decoding.

Authentication: API key in the X-OAI-API-KEY header. Set via
ORIGINALITY_API_KEY environment variable.

API CONTRACT NOTE:
    The Originality.ai documentation surface (docs.originality.ai)
    is JavaScript-rendered and the exact request/response schema
    could not be pinned to the byte from public docs at build time.
    The implementation below follows the most consistent contract
    described across third-party integration write-ups and the
    Originality help center page:

      Endpoint:   POST https://api.originality.ai/api/v1/scan/ai
      Headers:    X-OAI-API-KEY: <key>
                  Accept: application/json
                  Content-Type: application/json
      Body:       {"content": "<text>", "title": "<optional>",
                   "aiModelVersion": "turbo", "storeScan": false}
      Response:   {"score": {"ai": <0-1>, "original": <0-1>},
                   "credits_used": <int>,
                   "credits": <int>,
                   ... }   plus color-band-bearing fields.

    TODO[OPERATOR]: validate the exact field names against your
    account's API console before relying on this in production.
    The four call-sites flagged with TODO[OPERATOR] below are where
    field-name drift would surface. Once verified, delete the
    TODO[OPERATOR] markers and the function should run as-is.

Rate-limit note: Originality.ai documents 500 req/min on the
standard plan, with higher limits available on request. The
wrapper implements basic retry-on-429 with exponential backoff.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional, Tuple

import requests

logger = logging.getLogger("phase2.originality")


# ============================================================================
# >>> PASTE YOUR ORIGINALITY.AI API KEY HERE <<<
# ============================================================================
#
# Replace None with your key as a string, e.g.:
#     API_KEY = "sk-orig-abc123..."
#
# Leaving this as None falls back to the ORIGINALITY_API_KEY environment
# variable. Either path works; the inline constant is checked first.
# ============================================================================

API_KEY = None


# ============================================================================
# Constants
# ============================================================================

API_BASE = "https://api.originality.ai/api/v1"
# TODO[OPERATOR]: confirm the exact endpoint path. The /scan/ai
# path matches the v1 surface; v2 docs use /scan with a model
# parameter. If your account is on v2, change API_PATH to "/scan"
# and add an aiModelVersion="turbo" key to the body.
API_PATH = "/scan/ai"

# Originality.ai supports several detection models; the operator's
# Phase 1 workflow uses turbo. Standard turbo is approximately
# deterministic on identical text (operator-confirmed ±1 point),
# which is why Phase 2 does not re-score the same text.
DEFAULT_MODEL = "turbo"

DEFAULT_TIMEOUT_SEC = 60
MAX_RETRIES = 3
INITIAL_BACKOFF_SEC = 2.0


# ============================================================================
# Exceptions
# ============================================================================

class OriginalityAPIError(Exception):
    """Base class for Originality wrapper errors."""


class OriginalityAuthError(OriginalityAPIError):
    """API key missing or rejected."""


class OriginalityRateLimitError(OriginalityAPIError):
    """429 from the API; retries exhausted."""


class OriginalityResponseError(OriginalityAPIError):
    """API responded but the response was not parseable in the
    expected shape."""


# ============================================================================
# Public entry point
# ============================================================================

def score_text(
    text: str,
    title: Optional[str] = None,
    store_scan: bool = False,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    timeout: float = DEFAULT_TIMEOUT_SEC,
) -> Tuple[float, dict]:
    """
    Submit `text` to the Originality.ai AI-detection endpoint
    and return (human_probability_score_0_to_100, full_response_dict).

    The operator's working metric is the human-probability score
    (1 - AI probability) expressed as a 0-100 number. This wrapper
    converts the API's 0-1 "original" field to that scale.

    Parameters
    ----------
    text:        the chapter draft text to scan.
    title:       optional human-friendly label persisted with the scan.
    store_scan:  whether Originality should store the scan in the
                 account's history. Default False to keep the account
                 storage clean during automated runs.
    model:       detection model — "turbo" is Phase 2's default.
    api_key:     defaults to ORIGINALITY_API_KEY env var.
    timeout:     per-request timeout in seconds.

    Returns
    -------
    (score, response_dict) where score is in [0, 100].

    Raises
    ------
    OriginalityAuthError, OriginalityRateLimitError,
    OriginalityResponseError, OriginalityAPIError.
    """
    if api_key is None:
        api_key = API_KEY or os.environ.get("ORIGINALITY_API_KEY")
    if not api_key:
        raise OriginalityAuthError(
            "No Originality.ai API key found. Paste your key into the "
            "API_KEY constant near the top of originality_api.py, OR "
            "set the ORIGINALITY_API_KEY environment variable."
        )

    url = API_BASE + API_PATH

    # TODO[OPERATOR]: confirm body schema. Field names below follow
    # the most consistent contract across docs sources. If your
    # account expects different field names, change them here.
    body = {
        "content": text,
        "aiModelVersion": model,
        "storeScan": store_scan,
    }
    if title is not None:
        body["title"] = title

    headers = {
        # TODO[OPERATOR]: confirm header name. X-OAI-API-KEY is the
        # documented form on the v1 surface; some integrations use
        # "Authorization: Bearer <key>". Adjust if needed.
        "X-OAI-API-KEY": api_key,
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    response_json = _post_with_retry(url, headers, body, timeout)
    score = _extract_score(response_json)
    return score, response_json


# ============================================================================
# Internals
# ============================================================================

def _post_with_retry(
    url: str,
    headers: dict,
    body: dict,
    timeout: float,
) -> dict:
    backoff = INITIAL_BACKOFF_SEC
    last_exc: Optional[Exception] = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = requests.post(
                url, headers=headers, json=body, timeout=timeout,
            )
        except requests.RequestException as e:
            last_exc = e
            logger.warning("Originality request error (attempt %d): %s",
                           attempt + 1, e)
            if attempt < MAX_RETRIES:
                time.sleep(backoff)
                backoff *= 2
                continue
            raise OriginalityAPIError(
                f"Originality request failed after {MAX_RETRIES + 1} "
                f"attempts: {e}"
            ) from e

        if response.status_code == 401 or response.status_code == 403:
            raise OriginalityAuthError(
                f"Auth failed ({response.status_code}): {response.text[:300]}"
            )

        if response.status_code == 429:
            logger.warning("Originality 429 rate-limit (attempt %d): %s",
                           attempt + 1, response.text[:200])
            if attempt < MAX_RETRIES:
                # Respect Retry-After if present.
                retry_after = response.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else backoff
                time.sleep(wait)
                backoff *= 2
                continue
            raise OriginalityRateLimitError(
                f"Rate-limited after {MAX_RETRIES + 1} attempts."
            )

        if response.status_code >= 500:
            logger.warning("Originality 5xx (attempt %d): %s",
                           attempt + 1, response.status_code)
            if attempt < MAX_RETRIES:
                time.sleep(backoff)
                backoff *= 2
                continue
            raise OriginalityAPIError(
                f"Originality server error {response.status_code}: "
                f"{response.text[:300]}"
            )

        if not response.ok:
            raise OriginalityAPIError(
                f"Originality returned {response.status_code}: "
                f"{response.text[:300]}"
            )

        try:
            return response.json()
        except ValueError as e:
            raise OriginalityResponseError(
                f"Originality returned non-JSON: {response.text[:300]}"
            ) from e

    # Fallthrough — should be unreachable.
    raise OriginalityAPIError(
        f"Unexpected retry-loop exit. Last error: {last_exc}"
    )


def _extract_score(response_json: dict) -> float:
    """
    Convert the API's response into a 0-100 human-probability score.

    TODO[OPERATOR]: validate the field path against a real response.
    The contract assumed below — response_json["score"]["original"]
    in [0, 1] — is the most consistent shape across docs sources.
    Plausible alternatives:
      - response_json["score"] is a flat float in [0, 1] for "original"
      - response_json["original_score"] is the float
      - response_json["score"]["human"] (renamed in some accounts)

    Once verified, simplify the function and remove the fallbacks.
    """
    score_block = response_json.get("score")

    # Shape 1: nested {"score": {"original": 0.xx, "ai": 0.xx}}
    if isinstance(score_block, dict):
        if "original" in score_block:
            return _to_percent(score_block["original"])
        if "human" in score_block:
            return _to_percent(score_block["human"])
        if "ai" in score_block:
            return _to_percent(1.0 - float(score_block["ai"]))
        raise OriginalityResponseError(
            f"score block missing 'original'/'human'/'ai' field: "
            f"keys={list(score_block.keys())}"
        )

    # Shape 2: flat numeric "score" assumed to be the original probability
    if isinstance(score_block, (int, float)):
        return _to_percent(score_block)

    # Shape 3: alternative field names at top level.
    for alt in ("original_score", "human_score", "human_probability"):
        if alt in response_json and isinstance(
            response_json[alt], (int, float)
        ):
            return _to_percent(response_json[alt])

    raise OriginalityResponseError(
        f"Could not locate score in response. Top-level keys: "
        f"{list(response_json.keys())}"
    )


def _to_percent(value: float) -> float:
    """
    Normalise a 0-1 probability to 0-100, leaving 0-100 alone.

    Originality returns probabilities in [0, 1]; the operator's
    Phase 1 export pipeline rounds to 0-100. If a future API
    version emits 0-100 natively, this function still does the
    right thing.
    """
    v = float(value)
    if 0.0 <= v <= 1.0:
        v = v * 100.0
    return round(v, 1)


# ============================================================================
# Standalone smoke test (not invoked at import time)
# ============================================================================

def _smoke_test():
    """
    Manual smoke test. Run with:
      python -c "import originality_api; originality_api._smoke_test()"
    Requires ORIGINALITY_API_KEY to be set and a couple of credits
    on the account.
    """
    sample = (
        "The morning was already pulling itself together when she "
        "stepped out onto the porch. She had not slept well. The "
        "wolf in her chest had paced through every hour, which was "
        "the wolf's way of saying things she did not want to hear."
    )
    score, raw = score_text(sample, title="phase2_smoke_test")
    print(f"score (human probability, 0-100): {score}")
    print("full response:")
    print(json.dumps(raw, indent=2))


if __name__ == "__main__":
    _smoke_test()
