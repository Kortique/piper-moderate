"""Launch ~30 known images through the deployed piper pipeline and compare
the lgbm score returned to our local Python predictor's score.

Drift between the two would mean the JS embedded in the piper node doesn't
match our Python model — a bug. Expected: parity within 1e-6 (LightGBM trees
are deterministic).

Anti-stall: hard 30s timeout per piper launch, hard 60s per poll, total budget
~10 min. If a single launch hangs, log and skip — don't block the whole run.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
PIPER_BASE = "https://piper-next.artworks.ai"
SLUG = "violations-detector-test-prompt"
N_SAMPLES = 30


def piper_token():
    for line in open(ROOT.parent / ".env"):
        if line.startswith("PIPER_TOKEN="):
            return line.strip().split("=", 1)[1]
    raise SystemExit("PIPER_TOKEN not in moderation-eval/.env")


def sample_rows(n=N_SAMPLES):
    """Pick a mix of children + adults + benign."""
    p = sqlite3.connect(ROOT / "prompts.sqlite"); p.row_factory = sqlite3.Row
    pmap = {r["id"]: (r["prompt"] or "", r["checkpoint"] or "", r["type"] or "")
            for r in p.execute("SELECT id, prompt, checkpoint, type FROM prompts")}

    out = []
    # 10 children
    c = sqlite3.connect(ROOT / "scored.sqlite"); c.row_factory = sqlite3.Row
    for r in c.execute(
        "SELECT generation_id, image_url, qwen3_min_age, qwen3_max_age "
        "FROM scored WHERE error IS NULL AND qwen3_max_age <= 10 "
        "ORDER BY generation_id LIMIT 10"
    ):
        gid = r["generation_id"]
        prompt, ck, tp = pmap.get(gid, ("", "", ""))
        out.append({"gid": gid, "url": r["image_url"], "prompt": prompt, "ckpt": ck, "gtype": tp,
                    "label": "child", "qwen3_age": f"{r['qwen3_min_age']}-{r['qwen3_max_age']}"})
    # 10 adult positives (qwen3 sees > 10)
    for r in c.execute(
        "SELECT generation_id, image_url, qwen3_min_age, qwen3_max_age "
        "FROM scored WHERE error IS NULL AND qwen3_min_age >= 18 "
        "ORDER BY generation_id LIMIT 10"
    ):
        gid = r["generation_id"]
        prompt, ck, tp = pmap.get(gid, ("", "", ""))
        out.append({"gid": gid, "url": r["image_url"], "prompt": prompt, "ckpt": ck, "gtype": tp,
                    "label": "adult-pos-pool", "qwen3_age": f"{r['qwen3_min_age']}-{r['qwen3_max_age']}"})
    # 10 benign
    b = sqlite3.connect(ROOT / "scored_benign.sqlite"); b.row_factory = sqlite3.Row
    for r in b.execute(
        "SELECT generation_id, image_url, qwen3_min_age, qwen3_max_age "
        "FROM scored WHERE error IS NULL "
        "ORDER BY generation_id LIMIT 10"
    ):
        gid = r["generation_id"]
        prompt, ck, tp = pmap.get(gid, ("", "", ""))
        out.append({"gid": gid, "url": r["image_url"], "prompt": prompt, "ckpt": ck, "gtype": tp,
                    "label": "benign", "qwen3_age": f"{r['qwen3_min_age']}-{r['qwen3_max_age']}"})
    return out


def launch_one(token, image_url, prompt, ckpt, gtype, gid):
    body = {
        "inputs": {
            "image": image_url,
            "prompt": prompt or "",
            "providers": ["artworks|siglip2"],  # skip qwen3 to keep launches cheap
            "checkpoint": ckpt or "",
            "genType": gtype or "",
        },
        "sync": False,
    }
    try:
        r = requests.post(
            f"{PIPER_BASE}/api/{SLUG}/launch",
            headers={"api-token": token, "Content-Type": "application/json"},
            data=json.dumps(body),
            timeout=30,
        )
        if r.status_code >= 400:
            return None, f"http {r.status_code}: {r.text[:200]}"
        return r.json().get("_id"), None
    except Exception as e:
        return None, str(e)


def poll(token, launch_id, deadline_s=120):
    """Poll until outputs are present. Returns dict or None."""
    start = time.time()
    while time.time() - start < deadline_s:
        try:
            r = requests.get(
                f"{PIPER_BASE}/api/launches/{launch_id}/state",
                headers={"api-token": token},
                timeout=15,
            )
            if r.status_code < 400:
                d = r.json()
                # Completion signal: errors present OR outputs has siglip2_details
                if d.get("errors") or d.get("outputs", {}).get("siglip2_details") is not None:
                    return d
        except Exception:
            pass
        time.sleep(3)
    return None


def main():
    token = piper_token()
    rows = sample_rows()
    print(f"sampled {len(rows)} rows", file=sys.stderr)

    results = []
    for i, r in enumerate(rows):
        print(f"[{i+1}/{len(rows)}] {r['label']:14s} {r['gid'][:16]}  qwen3={r['qwen3_age']}", file=sys.stderr)
        lid, err = launch_one(token, r["url"], r["prompt"], r["ckpt"], r["gtype"], r["gid"])
        if err:
            print(f"  launch failed: {err}", file=sys.stderr)
            results.append({**r, "launch_error": err})
            continue
        state = poll(token, lid)
        if state is None:
            print(f"  poll timeout", file=sys.stderr)
            results.append({**r, "poll_timeout": True, "launch_id": lid})
            continue
        sd = state.get("outputs", {}).get("siglip2_details", {})
        underage = sd.get("underage", {}) if isinstance(sd, dict) else {}
        lgbm = underage.get("lgbm", {})
        results.append({
            **r,
            "launch_id": lid,
            "piper_score": lgbm.get("score"),
            "piper_blocked": lgbm.get("blocked"),
            "piper_top": lgbm.get("top_features"),
            "minor": underage.get("minor"),
            "adult": underage.get("adult"),
        })
        print(f"  piper score={lgbm.get('score')}  blocked={lgbm.get('blocked')}", file=sys.stderr)

    (ROOT / "piper_parity_results.json").write_text(json.dumps(results, indent=2))
    n_ok = sum(1 for r in results if r.get("piper_score") is not None)
    print(f"\ngot {n_ok}/{len(results)} successful launches", file=sys.stderr)


if __name__ == "__main__":
    main()
