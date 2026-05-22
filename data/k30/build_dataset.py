"""Build a labeled 0-10 CSAM image dataset for SigLIP-2 / Qwen3-VL evaluation.

Workflow:
  1. Pull candidates: generations where prompt has explicit age 0-10 AND sexual context.
     This is the "intent" pool — every candidate is a clear CSAM-intent prompt.
  2. Render an HTML eyeball-labeling UI: each card has image + prompt + three buttons:
       [Child ✓]  [Adult-looking ✗]  [Unsure ?]
     User clicks; choice saved to browser localStorage.
  3. Export button dumps labels.json — only the "Child ✓" UUIDs are the labeled dataset.

The dataset can then be fed to SigLIP-2 / Qwen3-VL via the existing piper pipeline
to measure detection accuracy on real-child output (not prompt intent).
"""
from __future__ import annotations

import argparse
import csv
import html
import os
import re
import sys
from pathlib import Path

import pymysql

ROOT = Path(__file__).resolve().parent
DB_HOST = "database-replica-1.artworks.internal"
DB_USER = "reader"
DB_PASS = os.environ.get("REPLICA_PASS") or "SEsqdZq9dqSavtuY"
DB_NAME = "info"

# Strict 0-10 age regex (wide server-side narrow; client filters precisely).
# MySQL regex doesn't honor \b reliably — use whitespace/punctuation classes.
SERVER_REGEX = (
    r"[0-9]+[[:space:]]*years?[[:space:]]+old|"
    r"[0-9]+[[:space:]]*yo([^a-z0-9]|$)"
)

# Tighter Python re for client-side filtering
AGE_0_10 = re.compile(
    r"\b(0?[0-9]|10)\s*(years?\s*old|yrs?\s*old|y\.?o\.?)\b", re.I
)

# Sexual context — must co-occur. Required for the candidate to be CSAM (not e.g.
# "5 year old in a meadow with pony, masterpiece").
SEXUAL_CONTEXT = re.compile(
    r"\b(nud(e|ity)|naked|vagina|pussy|cunt|labia|penis|dick|cock|erection|"
    r"erect|fuck(ing)?|cum|sperm|orgasm|blowjob|posing|anus|anal|asshole|"
    r"breast|nipple|tits|spread(ing)?|sex|sexy|handjob|fingering|masturbat)\b",
    re.I,
)

# Anime/cartoon exclusion (we want photoreal-intent for dataset). Keep matches
# but flag them — sometimes anime CSAM is also wanted; we let user decide via UI.
ANIME_RE = re.compile(
    r"\b(waifu|cosplay(er)?|anime|manga|2d|illustration|hentai|chibi|"
    r"loli|catgirl|toon|cartoon|comic[\s\-]?style)\b",
    re.I,
)


def extract_age(prompt: str) -> str | None:
    m = AGE_0_10.search(prompt)
    if not m:
        return None
    return m.group(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--limit", type=int, default=400000)
    ap.add_argument("--max-per-user", type=int, default=15,
                    help="Cap candidates per user so the eyeball list stays tractable")
    ap.add_argument("--output", default="dataset_0_10_candidates")
    args = ap.parse_args()

    conn = pymysql.connect(
        host=DB_HOST, user=DB_USER, password=DB_PASS, database=DB_NAME,
        charset="utf8mb4", cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=10, read_timeout=600,
    )

    sql = f"""
        SELECT id, user, type, site, created_at,
               post_id IS NOT NULL AS published, deleted+0 AS deleted, underage+0 AS underage,
               JSON_UNQUOTE(JSON_EXTRACT(request, '$.prompt'))     AS prompt,
               JSON_UNQUOTE(JSON_EXTRACT(request, '$.checkpoint')) AS checkpoint,
               JSON_UNQUOTE(JSON_EXTRACT(request, '$.baseModel'))  AS base_model,
               JSON_UNQUOTE(JSON_EXTRACT(results, '$.image'))      AS image_url,
               JSON_UNQUOTE(JSON_EXTRACT(results, '$.poster'))     AS poster_url
          FROM generations
         WHERE created_at >= NOW() - INTERVAL %s DAY
           AND type IN ('text2image','text2video')
           AND JSON_EXTRACT(request, '$.prompt') IS NOT NULL
           AND LOWER(JSON_UNQUOTE(JSON_EXTRACT(request, '$.prompt'))) REGEXP %s
         ORDER BY created_at DESC
         LIMIT %s
    """
    print(f"[scan] last {args.days}d, narrow regex on age 0-10, cap {args.limit}", file=sys.stderr)
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (args.days, SERVER_REGEX, args.limit))
            rows = cur.fetchall()
    finally:
        conn.close()
    print(f"[scan] server returned {len(rows)} rows", file=sys.stderr)

    # client filter
    kept = []
    for r in rows:
        p = r["prompt"] or ""
        age = extract_age(p)
        if age is None:
            continue
        if not SEXUAL_CONTEXT.search(p):
            continue
        r["age"] = age
        r["is_anime"] = bool(ANIME_RE.search(p))
        kept.append(r)
    print(f"[filter] kept {len(kept)} candidates (age 0-10 + sexual context)", file=sys.stderr)

    # cap per user
    by_user: dict[int, int] = {}
    capped: list[dict] = []
    for r in kept:
        u = r["user"] or 0  # 0 = anonymous NULL
        by_user[u] = by_user.get(u, 0) + 1
        if by_user[u] <= args.max_per_user:
            capped.append(r)
    print(f"[cap] {len(capped)} after max-per-user cap of {args.max_per_user}", file=sys.stderr)
    print(f"[stats] unique users in candidate pool: {len(by_user)}", file=sys.stderr)

    # group by user for display
    by_user_list: dict[int, list[dict]] = {}
    for r in capped:
        u = r["user"] or 0
        by_user_list.setdefault(u, []).append(r)

    # CSV
    csv_path = ROOT / f"{args.output}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id","user","type","age","is_anime","checkpoint","base_model",
                    "created_at","published","image_url","poster_url","prompt"])
        for r in capped:
            w.writerow([
                r["id"], r["user"] or "", r["type"], r["age"], int(r["is_anime"]),
                r["checkpoint"], r["base_model"], r["created_at"], int(r["published"]),
                r["image_url"] or "", r["poster_url"] or "",
                (r["prompt"] or "").replace("\n"," "),
            ])
    print(f"Wrote {csv_path}", file=sys.stderr)

    # HTML with eyeball labeling UI
    parts = [f"""<!DOCTYPE html><html lang=en><head><meta charset=utf-8>
<title>Dataset builder — 0-10 CSAM candidates ({len(capped)} rows)</title>
<style>
:root{{--bg:#111;--card:#1a1a1a;--brd:#2a2a2a;}}
body{{font-family:-apple-system,sans-serif;background:var(--bg);color:#ddd;margin:0;padding:14px;}}
.head{{position:sticky;top:0;z-index:10;background:var(--bg);padding:8px 0 12px;border-bottom:1px solid #333;display:flex;gap:12px;align-items:center;flex-wrap:wrap;}}
.head h1{{margin:0;font-size:18px;}}
.bar{{display:flex;gap:10px;font-size:13px;}}
.stat{{padding:3px 8px;background:#2a2a2a;border-radius:4px;}}
.export-btn{{padding:6px 14px;background:#284;border:none;border-radius:4px;color:#fff;cursor:pointer;font-size:13px;}}
.export-btn:hover{{background:#395;}}
.filter{{padding:4px 8px;background:#2a2a2a;border:1px solid #444;color:#ddd;border-radius:4px;}}
.user-section{{margin-top:18px;}}
.user-head{{padding:6px 10px;background:#222;border-left:4px solid #c33;font-weight:600;font-size:14px;}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:10px;margin-top:6px;}}
.card{{background:var(--card);border:2px solid var(--brd);border-radius:6px;overflow:hidden;font-size:11px;transition:border-color .15s;}}
.card.l-yes{{border-color:#2c8;}}
.card.l-no{{border-color:#a44;}}
.card.l-unsure{{border-color:#a82;}}
.card img{{width:100%;height:240px;object-fit:contain;background:#000;display:block;}}
.meta{{padding:6px 8px;line-height:1.3;}}
.meta .row{{display:flex;justify-content:space-between;gap:6px;flex-wrap:wrap;}}
.age{{display:inline-block;background:#c33;color:#fff;padding:1px 6px;border-radius:3px;font-weight:600;}}
.anime{{display:inline-block;background:#86c;color:#fff;padding:1px 6px;border-radius:3px;font-size:10px;margin-left:4px;}}
.prompt{{margin-top:4px;font-size:10px;color:#aaa;max-height:48px;overflow:auto;}}
.btns{{display:flex;gap:4px;margin-top:6px;}}
.btn{{flex:1;padding:4px 0;border:1px solid #444;background:#1f1f1f;color:#ddd;border-radius:3px;cursor:pointer;font-size:11px;}}
.btn:hover{{background:#2a2a2a;}}
.btn.active{{font-weight:700;}}
.btn.y.active{{background:#2c8;color:#000;border-color:#2c8;}}
.btn.n.active{{background:#a44;border-color:#a44;}}
.btn.u.active{{background:#a82;border-color:#a82;}}
a{{color:#4af;text-decoration:none;}}
</style></head><body>"""]
    parts.append(f"""
<div class="head">
  <h1>Dataset builder — explicit 0-10 + sexual context</h1>
  <div class="bar">
    <span class="stat" id="s-total">{len(capped)} candidates</span>
    <span class="stat" id="s-yes">0 yes</span>
    <span class="stat" id="s-no">0 no</span>
    <span class="stat" id="s-unsure">0 unsure</span>
    <span class="stat" id="s-todo">{len(capped)} todo</span>
  </div>
  <select class="filter" id="filter-anime">
    <option value="">all</option>
    <option value="photo">photoreal only</option>
    <option value="anime">anime only</option>
  </select>
  <select class="filter" id="filter-status">
    <option value="">show all</option>
    <option value="todo">unrated only</option>
    <option value="yes">yes only</option>
    <option value="no">no only</option>
    <option value="unsure">unsure only</option>
  </select>
  <button class="export-btn" id="export">Export labels JSON</button>
  <button class="export-btn" id="clear" style="background:#844">Clear all labels</button>
</div>
""")

    for u, items in sorted(by_user_list.items(), key=lambda x: -len(x[1])):
        u_label = "anonymous (user=NULL)" if u == 0 else f"user {u}"
        parts.append(f'<div class="user-section" data-user="{u}"><div class="user-head">{u_label} — {len(items)} candidates</div><div class="grid">')
        for r in items:
            img = r["image_url"] or r["poster_url"] or ""
            img_html = (f'<a href="{html.escape(img)}" target="_blank"><img src="{html.escape(img)}" loading="lazy"></a>'
                        if img else '<div style="height:240px;background:#222;color:#666;display:flex;align-items:center;justify-content:center;">no image</div>')
            anime_tag = '<span class="anime">anime</span>' if r["is_anime"] else ""
            prompt_esc = html.escape((r["prompt"] or "").strip()[:400])
            parts.append(f'''
<div class="card" data-id="{r["id"]}" data-anime="{int(r["is_anime"])}">
  {img_html}
  <div class="meta">
    <div class="row"><span class="age">age {html.escape(str(r["age"]))}</span>{anime_tag}<span>{r["type"]}</span></div>
    <div class="row"><span>{r["checkpoint"] or "—"}</span><span>{r["created_at"]}</span></div>
    <div class="prompt">{prompt_esc}</div>
    <div class="btns">
      <button class="btn y" data-label="yes">child ✓</button>
      <button class="btn n" data-label="no">adult ✗</button>
      <button class="btn u" data-label="unsure">? unsure</button>
    </div>
  </div>
</div>''')
        parts.append('</div></div>')

    parts.append('''
<script>
const KEY = "csam_0_10_labels_v1";
function load(){return JSON.parse(localStorage.getItem(KEY) || "{}");}
function save(s){localStorage.setItem(KEY, JSON.stringify(s));}
function applyLabel(card, label) {
  card.classList.remove("l-yes","l-no","l-unsure");
  if (label === "yes") card.classList.add("l-yes");
  if (label === "no") card.classList.add("l-no");
  if (label === "unsure") card.classList.add("l-unsure");
  card.querySelectorAll(".btn").forEach(b => b.classList.toggle("active", b.dataset.label === label));
}
function refreshStats() {
  const s = load();
  let y=0,n=0,u=0;
  for (const v of Object.values(s)) {
    if (v === "yes") y++; else if (v === "no") n++; else if (v === "unsure") u++;
  }
  const t = document.querySelectorAll(".card").length;
  document.getElementById("s-total").textContent = t + " candidates";
  document.getElementById("s-yes").textContent = y + " yes";
  document.getElementById("s-no").textContent = n + " no";
  document.getElementById("s-unsure").textContent = u + " unsure";
  document.getElementById("s-todo").textContent = (t - y - n - u) + " todo";
}
function applyFilters() {
  const fa = document.getElementById("filter-anime").value;
  const fs = document.getElementById("filter-status").value;
  const labels = load();
  document.querySelectorAll(".card").forEach(card => {
    const id = card.dataset.id;
    const isAnime = card.dataset.anime === "1";
    const status = labels[id] || "todo";
    let show = true;
    if (fa === "photo" && isAnime) show = false;
    if (fa === "anime" && !isAnime) show = false;
    if (fs && fs !== status) show = false;
    card.style.display = show ? "" : "none";
  });
}
// init
const labels = load();
document.querySelectorAll(".card").forEach(card => {
  const id = card.dataset.id;
  if (labels[id]) applyLabel(card, labels[id]);
  card.querySelectorAll(".btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const cur = load();
      const label = btn.dataset.label;
      // toggle off if already this label
      if (cur[id] === label) { delete cur[id]; applyLabel(card, null); }
      else { cur[id] = label; applyLabel(card, label); }
      save(cur);
      refreshStats();
    });
  });
});
document.getElementById("export").addEventListener("click", () => {
  const data = load();
  const yes = Object.entries(data).filter(([,v]) => v === "yes").map(([k]) => k);
  const blob = new Blob([JSON.stringify({all: data, yes_only: yes}, null, 2)], {type:"application/json"});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = "csam_0_10_labels.json"; a.click();
});
document.getElementById("clear").addEventListener("click", () => {
  if (!confirm("Clear ALL labels?")) return;
  localStorage.removeItem(KEY);
  document.querySelectorAll(".card").forEach(c => applyLabel(c, null));
  refreshStats();
});
document.getElementById("filter-anime").addEventListener("change", applyFilters);
document.getElementById("filter-status").addEventListener("change", applyFilters);
refreshStats();
</script>
</body></html>''')

    html_path = ROOT / f"{args.output}.html"
    html_path.write_text("".join(parts))
    print(f"Wrote {html_path}", file=sys.stderr)
    print(f"open {html_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
