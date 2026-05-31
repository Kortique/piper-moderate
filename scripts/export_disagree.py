"""
export_disagree.py
------------------
Fetch the latest N "disagree" reviews from stat.artworks.ai (Grafana/ClickHouse)
and append new records to data/disagree_pool.json.

Usage:
    python scripts/export_disagree.py              # last 500 disagree
    python scripts/export_disagree.py --limit 1000 --hours 72
    python scripts/export_disagree.py --dry-run    # show what would be added

Pool file: data/disagree_pool.json
  {
    "<generation_uuid>": {
      "id":          "<uuid>",
      "thumb_url":   "https://s3.realistic-media.io/.../thumbnail.webp",
      "prompt":      "...",
      "exported_at": "2026-05-17T12:00:00Z",
      "label":       null,          # child | teen | adult  (set in gallery)
      "labeled_at":  null,
      "piper_result": null          # filled by run_disagree_pipeline.py
    }
  }
"""

import os
import sys
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
import click
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn, DownloadColumn, TransferSpeedColumn

load_dotenv()
console = Console()

BASE_DIR    = Path(__file__).resolve().parent.parent
POOL_FILE   = BASE_DIR / "data" / "disagree_pool.json"
IMAGES_DIR  = BASE_DIR / "data" / "disagree_images"

GRAFANA_BASE    = "https://stat.artworks.ai"
GRAFANA_SESSION = os.getenv("GRAFANA_SESSION", "")
CH_DATASOURCE   = "d2aa1ac6-ddb1-4a64-bc99-eba5f748298b"


def _get_session_cookie() -> str:
    """Return a valid grafana_session, refreshing via SSO if needed."""
    cookie = GRAFANA_SESSION
    if cookie:
        # Quick check
        r = httpx.get(
            f"{GRAFANA_BASE}/api/user",
            headers={"Cookie": f"grafana_session={cookie}"},
            timeout=10,
        )
        if r.status_code == 200:
            return cookie
    # Session expired or missing — re-login
    console.print("[yellow]Grafana session expired, re-authenticating...[/yellow]")
    from scripts.grafana_login import get_grafana_session
    from dotenv import set_key
    new_cookie = get_grafana_session()
    # Persist to .env so next run reuses it
    env_path = str(BASE_DIR / ".env")
    set_key(env_path, "GRAFANA_SESSION", new_cookie)
    console.print(f"[green]New session obtained and saved to .env[/green]")
    return new_cookie


def grafana_query(sql: str) -> list[dict]:
    """Execute a ClickHouse query via Grafana DS proxy, auto-refreshing session."""
    cookie = _get_session_cookie()
    headers = {
        "Cookie": f"grafana_session={cookie}",
        "Content-Type": "application/json",
        "X-Grafana-Org-Id": "1",
    }
    payload = {
        "queries": [{
            "refId": "A",
            "datasource": {"uid": CH_DATASOURCE, "type": "grafana-clickhouse-datasource"},
            "rawSql": sql,
            "format": 1,
        }],
        "from": "now-30d",
        "to": "now",
    }
    with httpx.Client(timeout=30) as client:
        r = client.post(f"{GRAFANA_BASE}/api/ds/query", headers=headers, json=payload)
        r.raise_for_status()

    result = r.json()["results"]["A"]
    if result.get("error"):
        raise RuntimeError(f"ClickHouse error: {result['error']}")

    frames = result.get("frames", [])
    if not frames:
        return []

    frame = frames[0]
    cols  = [f["name"] for f in frame["schema"]["fields"]]
    rows  = list(zip(*frame["data"]["values"]))
    return [dict(zip(cols, row)) for row in rows]


try:
    import fcntl as _fcntl_mod
except ImportError:
    _fcntl_mod = None
import signal as _signal_mod

_POOL_LOCK_FILE = str(POOL_FILE) + ".lock"
_pool_lock_fh = None

def _try_lock_pool():
    """Best-effort advisory lock so two writers can't race on POOL_FILE."""
    global _pool_lock_fh
    if _fcntl_mod is None:
        return True
    try:
        _pool_lock_fh = open(_POOL_LOCK_FILE, "w")
        _fcntl_mod.flock(_pool_lock_fh.fileno(), _fcntl_mod.LOCK_EX | _fcntl_mod.LOCK_NB)
        _pool_lock_fh.write(str(os.getpid()))
        _pool_lock_fh.flush()
        return True
    except (OSError, BlockingIOError):
        return False


def load_pool() -> dict:
    if POOL_FILE.exists():
        try:
            raw = POOL_FILE.read_bytes().rstrip(b'\x00')
            return json.loads(raw)
        except Exception as e:
            tmp = POOL_FILE.parent / (POOL_FILE.name + ".tmp")
            if tmp.exists():
                try:
                    console.print(f"[yellow]WARN: pool corrupt ({e}); falling back to .tmp[/yellow]")
                    return json.loads(tmp.read_bytes().rstrip(b'\x00'))
                except Exception:
                    pass
            raise
    return {}


def save_pool(pool: dict):
    """Atomic + signal-safe pool write."""
    POOL_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        old_term = _signal_mod.signal(_signal_mod.SIGTERM, _signal_mod.SIG_IGN)
    except Exception:
        old_term = None
    try:
        old_int = _signal_mod.signal(_signal_mod.SIGINT, _signal_mod.SIG_IGN)
    except Exception:
        old_int = None
    try:
        tmp = str(POOL_FILE) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(pool, f, indent=2, ensure_ascii=False)
            f.flush()
            try:
                os.fsync(f.fileno())
            except (OSError, AttributeError):
                pass
        os.replace(tmp, POOL_FILE)
    finally:
        if old_term is not None:
            try: _signal_mod.signal(_signal_mod.SIGTERM, old_term)
            except Exception: pass
        if old_int is not None:
            try: _signal_mod.signal(_signal_mod.SIGINT, old_int)
            except Exception: pass


def download_image(gen_id: str, url: str) -> tuple[str, str | None]:
    """Download one thumbnail. Returns (gen_id, local_path | None)."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    # Determine extension from URL (usually .webp)
    ext = Path(url.split("?")[0]).suffix or ".webp"
    dest = IMAGES_DIR / f"{gen_id}{ext}"
    if dest.exists():
        return gen_id, str(dest.relative_to(BASE_DIR))
    try:
        with httpx.Client(timeout=20, follow_redirects=True) as client:
            r = client.get(url)
            r.raise_for_status()
            dest.write_bytes(r.content)
        return gen_id, str(dest.relative_to(BASE_DIR))
    except Exception as e:
        console.print(f"[red]  ✗ {gen_id[:8]}: {e}[/red]")
        return gen_id, None


def download_missing(pool: dict, workers: int = 8) -> int:
    """Download images not yet on disk. Returns count of newly downloaded."""
    to_download = [
        (v["id"], v["thumb_url"])
        for v in pool.values()
        if not v.get("local_path") or not (BASE_DIR / v["local_path"]).exists()
    ]
    if not to_download:
        console.print("[dim]All images already downloaded.[/dim]")
        return 0

    console.print(f"[dim]Downloading {len(to_download)} images ({workers} threads)...[/dim]")

    downloaded = 0
    failed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Downloading...", total=len(to_download))

        with ThreadPoolExecutor(max_workers=workers) as exe:
            futures = {exe.submit(download_image, gid, url): gid for gid, url in to_download}
            for future in as_completed(futures):
                gen_id, local_path = future.result()
                if local_path:
                    pool[gen_id]["local_path"] = local_path
                    downloaded += 1
                else:
                    failed += 1
                progress.advance(task)

    console.print(f"  downloaded={downloaded}  failed={failed}")
    return downloaded


@click.command()
@click.option("--limit",     default=500,  show_default=True, help="Max records to fetch from Grafana")
@click.option("--hours",     default=168,  show_default=True, help="Look back N hours (default: 7 days)")
@click.option("--dry-run",   is_flag=True, help="Do not write anything, just print stats")
@click.option("--no-download", is_flag=True, help="Skip image download (store URLs only)")
@click.option("--workers",   default=8,    show_default=True, help="Parallel download threads")
def main(limit, hours, dry_run, no_download, workers):
    """Fetch disagree reviews from Grafana, append to disagree_pool.json, download images."""

    # Single timestamp for all records added in this run
    export_batch = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    console.print(f"[dim]Fetching up to {limit} disagree reviews from last {hours}h...[/dim]")
    console.print(f"[dim]Export batch: {export_batch}[/dim]")

    sql = f"""
SELECT id, type, created_at, details
FROM generation_reviews
WHERE opinion = 'disagree'
  AND created_at >= now() - INTERVAL {hours} HOUR
  AND details != '{{}}'
  AND details != ''
ORDER BY created_at DESC
LIMIT {limit}
"""

    rows = grafana_query(sql.strip())
    console.print(f"[dim]  Got {len(rows)} rows from ClickHouse[/dim]")

    pool = load_pool()
    existing_ids = set(pool.keys())
    deleted_ids  = {k for k, v in pool.items() if v.get("deleted")}

    added   = 0
    skipped = 0
    no_image = 0

    for row in rows:
        gen_id = row.get("id")
        if not gen_id:
            continue

        details_raw = row.get("details") or ""
        try:
            details = json.loads(details_raw) if isinstance(details_raw, str) else details_raw
        except Exception:
            details = {}

        thumb_url = details.get("image") or ""
        if not thumb_url:
            no_image += 1
            continue

        if gen_id in deleted_ids:
            skipped += 1
            continue

        if gen_id in existing_ids:
            skipped += 1
            continue

        # Timestamp: Grafana returns ms epoch
        ts_ms = row.get("created_at")
        try:
            exported_dt = datetime.fromtimestamp(int(ts_ms) / 1000, tz=timezone.utc).isoformat()
        except Exception:
            exported_dt = datetime.now(timezone.utc).isoformat()

        pool[gen_id] = {
            "id":           gen_id,
            "thumb_url":    thumb_url,
            "prompt":       details.get("prompt", ""),
            "exported_at":  exported_dt,
            "export_batch": export_batch,
            "label":        None,
            "labeled_at":   None,
            "piper_result": None,
        }
        added += 1

    console.print(f"[bold]Results:[/bold]  added={added}  skipped(already in pool)={skipped}  no_image={no_image}")
    console.print(f"[bold]Pool total:[/bold] {len(pool)} records")

    if dry_run:
        console.print("[yellow]--dry-run: not saving[/yellow]")
        return

    if added > 0:
        save_pool(pool)
        console.print(f"[green]Saved → {POOL_FILE}[/green]")
    else:
        console.print("[dim]Nothing new to save.[/dim]")

    # Download images
    if not no_download:
        newly_downloaded = download_missing(pool, workers=workers)
        if newly_downloaded > 0:
            save_pool(pool)  # save updated local_path fields
            console.print(f"[green]Images → {IMAGES_DIR}[/green]")
    else:
        console.print("[dim]--no-download: skipping image download.[/dim]")

    # Print label distribution
    labels = [v["label"] for v in pool.values()]
    n_unlabeled = labels.count(None)
    n_child     = labels.count("child")
    n_teen      = labels.count("teen")
    n_adult     = labels.count("adult")
    console.print(
        f"\n[dim]Label stats:[/dim]  "
        f"unlabeled=[yellow]{n_unlabeled}[/yellow]  "
        f"child=[red]{n_child}[/red]  "
        f"teen=[yellow]{n_teen}[/yellow]  "
        f"adult=[green]{n_adult}[/green]"
    )

    # Auto-moderate newly added images (Qwen3 + Piper)
    if added > 0 and not dry_run:
        console.print(f"\n[dim]Running auto-moderation on {added} new images…[/dim]")
        try:
            from scripts.moderate_disagree import main as moderate_main
            from click.testing import CliRunner
            runner = CliRunner()
            result = runner.invoke(moderate_main, ["--workers", "3"], catch_exceptions=False)
            if result.output:
                console.print(result.output)
        except Exception as e:
            console.print(f"[yellow]Auto-moderation error: {e}[/yellow]")
            console.print("[dim]Run manually: python scripts/moderate_disagree.py[/dim]")


if __name__ == "__main__":
    main()
