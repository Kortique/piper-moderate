"""Score candidate images through Qwen3-VL (age estimate) and SigLIP-2 (underage risk)
via the piper CLAUDE TEST pipeline (slug=violations-detector-test-claude).

Resumable: writes to scored.sqlite keyed by generation_id.

Usage:
    python score_dataset.py --csv dataset_0_10_candidates.csv --limit 200 --concurrency 5
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter

ROOT = Path(__file__).resolve().parent
ENV_PATH = ROOT.parent / ".env"
DB_PATH = ROOT / "scored.sqlite"
PIPELINE_SLUG = "violations-detector-test-v3"


def load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    for k, v in os.environ.items():
        env.setdefault(k, v)
    return env


def piper_session(env):
    s = requests.Session()
    s.headers["api-token"] = env["PIPER_TOKEN"]
    s.headers["content-type"] = "application/json"
    adapter = HTTPAdapter(pool_connections=50, pool_maxsize=50)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


def init_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS scored (
            generation_id  TEXT PRIMARY KEY,
            image_url      TEXT NOT NULL,
            prompt_age     TEXT,
            user_id        INTEGER,
            qwen3_min_age  INTEGER,   -- min(faces[].ageFrom)
            qwen3_max_age  INTEGER,   -- max(faces[].ageTo)
            qwen3_underage INTEGER,   -- bool 0/1
            qwen3_nsfw     TEXT,
            qwen3_style    TEXT,
            qwen3_status   TEXT,
            qwen3_desc     TEXT,
            siglip_minor   REAL,
            siglip_adult   REAL,
            siglip_conf    REAL,
            raw_qwen3      TEXT,
            raw_siglip     TEXT,
            error          TEXT,
            scored_at      TEXT NOT NULL DEFAULT (datetime('now'))
        );
    """)
    return conn


def launch_both(s: requests.Session, base_url: str, image_url: str) -> str | None:
    """Single launch with both providers — cheaper, faster, atomic.
    providers must be a pipe-separated STRING (not an array) — the pipeline's
    prepare_params node splits on '|'. Must prefix with 'artworks' to enable
    artworks-platform output. Verified working: artworks|siglip2|qwen3 → both
    siglip2_details + qwen3_details populated in ~10-15s.
    """
    body = {
        "inputs": {
            "image": image_url,
            "providers": "artworks|siglip2|qwen3",
            "prompt": "",
            "config": "{}",
        }
    }
    r = s.post(f"{base_url}/api/{PIPELINE_SLUG}/launch", data=json.dumps(body), timeout=30)
    r.raise_for_status()
    return r.json().get("_id")


def poll(s: requests.Session, base_url: str, launch_id: str, deadline_s: float) -> dict | None:
    """Wait until BOTH siglip2_details and qwen3_details appear, or errors are
    present, or timeout. The pipeline progressively populates outputs — early
    returns can miss the slower (deferred) provider."""
    while time.monotonic() < deadline_s:
        r = s.get(f"{base_url}/api/launches/{launch_id}/state", timeout=30)
        r.raise_for_status()
        d = r.json()
        out = d.get("outputs") or {}
        errs = d.get("errors") or []
        if errs:
            return d
        if out.get("siglip2_details") and out.get("qwen3_details"):
            return d
        time.sleep(1.5)
    return None


def score_one(env: dict, lock: threading.Lock, conn: sqlite3.Connection, row: dict) -> dict:
    gen_id = row["id"]
    img = row["image_url"] or row["poster_url"] or ""
    if not img:
        return {"id": gen_id, "skipped": "no_image"}
    s = piper_session(env)
    deadline = time.monotonic() + 180
    try:
        lid = launch_both(s, env["PIPER_BASE_URL"], img)
        state = poll(s, env["PIPER_BASE_URL"], lid, deadline)
        out = (state or {}).get("outputs") or {}
        qw = out.get("qwen3_details") or {}
        sl = out.get("siglip2_details") or {}
        sl_under = sl.get("underage") or {}

        faces = qw.get("faces") or []
        qw_min_age = min((f.get("ageFrom") for f in faces if isinstance(f.get("ageFrom"), int)), default=None)
        qw_max_age = max((f.get("ageTo")   for f in faces if isinstance(f.get("ageTo"), int)),   default=None)

        with lock:
            conn.execute(
                """
                INSERT OR REPLACE INTO scored
                (generation_id, image_url, prompt_age, user_id,
                 qwen3_min_age, qwen3_max_age, qwen3_underage, qwen3_nsfw,
                 qwen3_style, qwen3_status, qwen3_desc,
                 siglip_minor, siglip_adult, siglip_conf,
                 raw_qwen3, raw_siglip, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    gen_id, img, row.get("age"), int(row["user"]) if row.get("user") else None,
                    qw_min_age, qw_max_age,
                    1 if qw.get("underage") else 0,
                    qw.get("nsfw"), qw.get("style"), qw.get("status"),
                    (qw.get("description") or "")[:500],
                    sl_under.get("minor"), sl_under.get("adult"), sl_under.get("confidence"),
                    json.dumps(qw), json.dumps(sl),
                ),
            )
            conn.commit()
        return {"id": gen_id, "qwen3_min_age": qw_min_age, "qwen3_underage": bool(qw.get("underage")),
                "siglip_minor": sl_under.get("minor")}
    except Exception as e:
        with lock:
            conn.execute(
                "INSERT OR REPLACE INTO scored (generation_id, image_url, prompt_age, user_id, error) VALUES (?, ?, ?, ?, ?)",
                (gen_id, img, row.get("age"), int(row["user"]) if row.get("user") else None, str(e)[:500]),
            )
            conn.commit()
        return {"id": gen_id, "error": str(e)[:300]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="dataset CSV produced by build_dataset.py")
    ap.add_argument("--limit", type=int, default=200, help="max rows to score this run")
    ap.add_argument("--concurrency", type=int, default=5)
    ap.add_argument("--skip-anime", action="store_true", help="skip rows where is_anime=1")
    ap.add_argument("--db", default=str(DB_PATH), help="sqlite output path")
    args = ap.parse_args()

    env = load_env()
    conn = init_db(Path(args.db))
    lock = threading.Lock()

    # already-scored ids (resume)
    done = {row[0] for row in conn.execute("SELECT generation_id FROM scored WHERE error IS NULL")}
    print(f"[resume] {len(done)} already scored", file=sys.stderr)

    # load CSV
    rows: list[dict] = []
    with open(args.csv) as f:
        for r in csv.DictReader(f):
            if r["id"] in done:
                continue
            if args.skip_anime and r.get("is_anime") == "1":
                continue
            rows.append(r)
            if len(rows) >= args.limit:
                break
    print(f"[scan] {len(rows)} new rows to score", file=sys.stderr)

    t0 = time.monotonic()
    n_ok = n_err = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(score_one, env, lock, conn, r) for r in rows]
        for i, f in enumerate(as_completed(futs), 1):
            res = f.result()
            if "error" in res:
                n_err += 1
                print(f"  [{i}/{len(rows)}] ERR {res['id'][:8]} {res['error'][:80]}", file=sys.stderr)
            else:
                n_ok += 1
                qa = res.get("qwen3_min_age"); qu = res.get("qwen3_underage"); sm = res.get("siglip_minor")
                print(f"  [{i}/{len(rows)}] ok {res['id'][:8]}  qwen3_min_age={qa}  qwen3_underage={qu}  siglip_minor={sm}", file=sys.stderr)
    elapsed = time.monotonic() - t0
    print(f"\n[done] ok={n_ok} err={n_err} in {elapsed:.0f}s ({elapsed/max(1,len(rows)):.2f}s/row)", file=sys.stderr)


if __name__ == "__main__":
    main()
