#!/usr/bin/env python3
"""
run_eval_batch.py — run items through Piper v4, save scores to results file.
Results saved incrementally. Safe to restart (resumes from where left off).

Usage:
    python scripts/run_eval_batch.py data/eval_g1_runs.json data/eval_g1_results.json
    python scripts/run_eval_batch.py data/eval_g2_runs.json data/eval_g2_results.json
"""
import os, sys, json, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv
import httpx

load_dotenv()

PIPER_BASE = "https://piper-next.artworks.ai/api"
PROJECT    = "d2911d10bb"
TOKEN      = os.getenv("PIPER_TOKEN", "")
WORKERS    = 6

def hdr():
    return {"User-Token": TOKEN, "Content-Type": "application/json", "Accept": "application/json"}

def run_one(item):
    try:
        r = httpx.post(f"{PIPER_BASE}/projects/{PROJECT}/launch",
            headers=hdr(), json={"inputs": {"image": item['url']}}, timeout=30)
        r.raise_for_status()
        run_id = r.json()["_id"]
        for _ in range(40):
            time.sleep(3)
            rs = httpx.get(f"{PIPER_BASE}/launches/{run_id}/state", headers=hdr(), timeout=15)
            if rs.status_code != 200: continue
            state = rs.json()
            outputs = state.get("outputs") or {}
            if "siglip2_labels" in outputs:
                det  = outputs.get("siglip2_details", {}).get("underage", {})
                lgbm = det.get("lgbm") or {}
                return {**item, "run_id": run_id,
                        "minor": det.get("minor", 0),
                        "lgbm":  lgbm.get("score", 0),
                        "blocked": outputs.get("blocked", False),
                        "done": True}
        return {**item, "error": "timeout"}
    except Exception as e:
        return {**item, "error": str(e)}

def main():
    if len(sys.argv) < 3:
        print("Usage: run_eval_batch.py <input.json> <output.json>")
        sys.exit(1)

    inp  = Path(sys.argv[1])
    out  = Path(sys.argv[2])

    items = json.loads(inp.read_text())

    # Resume
    existing = {}
    if out.exists():
        for r in json.loads(out.read_text()):
            if r.get('done') or r.get('error') == 'no_url':
                existing[r['id']] = r

    todo = [x for x in items if x['id'] not in existing]
    print(f"Total: {len(items)} | Already done: {len(existing)} | Todo: {len(todo)}")

    results = list(existing.values())

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(run_one, item): item for item in todo}
        n = len(existing)
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            n += 1
            ok = "✓" if res.get("done") else f"✗({res.get('error','?')[:20]})"
            print(f"  [{n:3d}/{len(items)}] {ok} {res['id']:20} "
                  f"lgbm={res.get('lgbm',0):.3f} label={res.get('human_label','?')}")
            out.write_text(json.dumps(results, indent=2))

    done = sum(1 for r in results if r.get('done'))
    errs = sum(1 for r in results if r.get('error'))
    print(f"\nFinished: {done} ok, {errs} errors out of {len(items)}")

if __name__ == "__main__":
    main()
