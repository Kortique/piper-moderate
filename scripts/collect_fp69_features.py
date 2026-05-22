"""
collect_fp69_features.py
------------------------
Run 69 FP images through Piper and save full siglip2_details.underage per-tag scores.
Used to prepare training data for LGBM retraining.

Usage:
    python scripts/collect_fp69_features.py
"""
import os, sys, json, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import httpx
from dotenv import load_dotenv

load_dotenv()

BASE_DIR   = Path(__file__).resolve().parent.parent
DB_PATH    = BASE_DIR / "gallery.db"
FP_IDS     = BASE_DIR / "fp_69_ids.txt"

PIPER_BASE = "https://piper-next.artworks.ai/api"
PROJECT    = "d2911d10bb"
TOKEN      = os.getenv("PIPER_TOKEN", "")
WORKERS    = 4

def hdr():
    return {"User-Token": TOKEN, "Content-Type": "application/json", "Accept": "application/json"}

def run_one(item):
    try:
        r = httpx.post(f"{PIPER_BASE}/projects/{PROJECT}/launch",
            headers=hdr(), json={"inputs": {"image": item["url"]}}, timeout=30)
        r.raise_for_status()
        run_id = r.json()["_id"]
        for _ in range(40):
            time.sleep(3)
            rs = httpx.get(f"{PIPER_BASE}/launches/{run_id}/state", headers=hdr(), timeout=15)
            if rs.status_code != 200: continue
            state = rs.json()
            outputs = state.get("outputs") or {}
            if "siglip2_labels" in outputs:
                det = outputs.get("siglip2_details", {}).get("underage", {})
                lgbm_o = det.get("lgbm") or {}
                adult_labels = det.get("labels", {}).get("adult") or {}
                underage_labels = det.get("labels", {}).get("underage") or {}
                return {
                    "id": item["id"],
                    "url": item["url"],
                    "label": "adult",
                    "lgbm_score": lgbm_o.get("score", 0),
                    "minor_score": det.get("minor", 0),
                    "adult_score": det.get("adult", 0),
                    "lgbm_blocked": lgbm_o.get("blocked", False),
                    "underage_labels": underage_labels,
                    "adult_labels": adult_labels,
                    "done": True
                }
        return {"id": item["id"], "error": "timeout"}
    except Exception as e:
        return {"id": item["id"], "error": str(e)}

def main():
    import sqlite3
    fp_ids = [l.strip() for l in FP_IDS.read_text().splitlines() if l.strip()]

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = lambda c, r: dict(zip([col[0] for col in c.description], r))

    items = []
    for gid in fp_ids:
        row = conn.execute("SELECT thumb_url FROM grafana_pool WHERE id=?", (gid,)).fetchone()
        if row and row["thumb_url"]:
            items.append({"id": gid, "url": row["thumb_url"]})
    conn.close()

    print(f"Processing {len(items)} FP images...")
    results = []

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(run_one, item): item for item in items}
        n = 0
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            n += 1
            ok = "✓" if res.get("done") else f"✗({res.get('error','?')[:20]})"
            print(f"  [{n:3d}/{len(items)}] {ok} {res['id'][:25]} lgbm={res.get('lgbm_score',0):.3f}")

    out = BASE_DIR / "data" / "fp69_features_new_tags.json"
    out.write_text(json.dumps([r for r in results if r.get("done")], indent=2, ensure_ascii=False))
    done = sum(1 for r in results if r.get("done"))
    print(f"\nDone: {done}/{len(items)} saved to {out}")

if __name__ == "__main__":
    main()
