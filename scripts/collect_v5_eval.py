"""
collect_v5_eval.py
------------------
Run eval groups through Piper (with updated tags already deployed),
capture full siglip2_details.underage.labels, apply V5 LGBM locally,
and save results in eval JSON format for gallery_server.

Usage:
    python scripts/collect_v5_eval.py --groups 1,2
    python scripts/collect_v5_eval.py --groups 1
"""
import os, sys, json, re, math, time, datetime, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import httpx
from dotenv import load_dotenv

load_dotenv()

BASE_DIR   = Path(__file__).resolve().parent.parent
PIPER_BASE = "https://piper-next.artworks.ai/api"
PROJECT    = "d2911d10bb"
TOKEN      = os.getenv("PIPER_TOKEN", "")
WORKERS    = 6

# ── V5 LGBM (compact format from lgbm_evaluate_v5_piper.js) ───────────────

def _load_v5_model():
    js_path = BASE_DIR / "data" / "lgbm_evaluate_v5_piper.js"
    js = js_path.read_text(encoding="utf-8")
    compact_trees = json.loads(re.search(r'const LGBM_TREES = (\[.*?\]);', js, re.DOTALL).group(1))
    compact_features = json.loads(re.search(r'const LGBM_FEATURES = (\[.*?\]);', js).group(1))
    return compact_trees, compact_features

LGBM_TREES, LGBM_FEATURES = _load_v5_model()
FEAT_IDX = {f: i for i, f in enumerate(LGBM_FEATURES)}
LGBM_THRESHOLD = 0.80


def v5_predict(underage_labels: dict, adult_labels: dict) -> float:
    """Apply V5 LGBM to per-tag score dicts."""
    vec = [0.0] * len(LGBM_FEATURES)
    for k, v in underage_labels.items():
        if k in FEAT_IDX:
            vec[FEAT_IDX[k]] = float(v)
    for k, v in adult_labels.items():
        fk = f"adult__{k}"
        if fk in FEAT_IDX:
            vec[FEAT_IDX[fk]] = float(v)
    # Tree traversal
    score = 0.0
    for t in LGBM_TREES:
        node = t["r"]
        while node >= 0:
            fi, thr = t["s"][node]
            l, r = t["c"][node]
            node = l if vec[fi] <= thr else r
        score += t["l"][-(node + 1)]
    return 1.0 / (1.0 + math.exp(-score))


# ── Piper API ───────────────────────────────────────────────────────────────

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
            if rs.status_code != 200:
                continue
            state = rs.json()
            outputs = state.get("outputs") or {}
            if "siglip2_labels" in outputs:
                det = outputs.get("siglip2_details", {}).get("underage", {})
                labels = det.get("labels", {})
                underage_labels = labels.get("underage") or {}
                adult_labels    = labels.get("adult") or {}
                minor_score     = det.get("minor", 0)
                # Compute V5 LGBM locally
                lgbm_v5 = v5_predict(underage_labels, adult_labels)
                return {
                    "id":     item["id"],
                    "url":    item["url"],
                    "minor":  minor_score,
                    "lgbm":   lgbm_v5,
                    "blocked": (lgbm_v5 >= LGBM_THRESHOLD) or (minor_score >= 0.72),
                    "human_label": item.get("human_label"),
                    "variant": item.get("variant"),
                    "done":   True,
                }
        return {"id": item["id"], "error": "timeout"}
    except Exception as e:
        return {"id": item["id"], "error": str(e)}


def load_group(group_n: int):
    """Load eval_gN_runs.json — human_label is already present in that file."""
    runs_path = BASE_DIR / "data" / f"eval_g{group_n}_runs.json"
    if not runs_path.exists():
        raise FileNotFoundError(f"Missing: {runs_path}")
    items = json.loads(runs_path.read_text())
    return items


def run_group(group_n: int, out_path: Path, resume: bool):
    items = load_group(group_n)
    print(f"Group {group_n}: {len(items)} images")

    # Resume: load existing results
    done_ids = set()
    existing = []
    if resume and out_path.exists():
        existing = json.loads(out_path.read_text())
        done_ids = {r["id"] for r in existing if r.get("done")}
        print(f"  Resuming: {len(done_ids)} already done")

    todo = [it for it in items if it["id"] not in done_ids]
    print(f"  Running {len(todo)} images...")

    results = list(existing)
    n = 0

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(run_one, item): item for item in todo}
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            n += 1
            ok = "✓" if res.get("done") else f"✗({res.get('error','?')[:20]})"
            lgbm_str = f"lgbm={res.get('lgbm', 0):.3f}" if res.get("done") else ""
            print(f"  [{n:4d}/{len(todo)}] {ok} {res['id'][:28]} {lgbm_str}")

            # Save incrementally every 20 results
            if n % 20 == 0:
                out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    # Final save
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    done = sum(1 for r in results if r.get("done"))
    print(f"  Saved {done}/{len(items)} to {out_path}")

    # Quick accuracy summary
    done_items = [r for r in results if r.get("done") and r.get("human_label")]
    minors = [r for r in done_items if r["human_label"] in ("child", "teen")]
    adults = [r for r in done_items if r["human_label"] == "adult"]
    if minors:
        recall = sum(1 for r in minors if r["blocked"]) / len(minors)
        print(f"  Recall (minor): {recall:.3f} ({sum(1 for r in minors if r['blocked'])}/{len(minors)})")
    if adults:
        fpr = sum(1 for r in adults if r["blocked"]) / len(adults)
        print(f"  FPR   (adult):  {fpr:.3f} ({sum(1 for r in adults if r['blocked'])}/{len(adults)})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", default="1,2", help="Comma-separated group numbers e.g. 1,2")
    parser.add_argument("--resume", action="store_true", help="Skip already processed images")
    parser.add_argument("--output", default=None, help="Output file path (default: data/eval_v5_g<N>_results.json)")
    args = parser.parse_args()

    group_nums = [int(g.strip()) for g in args.groups.split(",")]

    if len(group_nums) == 1:
        # Single group output
        gn = group_nums[0]
        out = Path(args.output) if args.output else BASE_DIR / "data" / f"eval_v5_g{gn}_results.json"
        run_group(gn, out, args.resume)
    else:
        # Multiple groups → one combined output
        all_results = []
        for gn in group_nums:
            tmp = BASE_DIR / "data" / f"_v5_g{gn}_tmp.json"
            run_group(gn, tmp, args.resume)
            all_results.extend(json.loads(tmp.read_text()))
            tmp.unlink(missing_ok=True)

        out = Path(args.output) if args.output else BASE_DIR / "data" / "eval_v5_g1g2_results.json"
        out.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
        print(f"\nCombined output: {out} ({len(all_results)} items)")


if __name__ == "__main__":
    main()
