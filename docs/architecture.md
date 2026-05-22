# Architecture

## System components

```
┌─────────────────────────────────────────────────────────────────────┐
│  Your machine                                                        │
│                                                                      │
│  piper-moderate/                                                     │
│  ├── scripts/run_category.py  ──────────────────►  n8n agent        │
│  │                                                  (webhook)        │
│  ├── scripts/analyze_failed.py  ──────────────►  OpenRouter         │
│  │       reads ◄── results/*.json                  (Grok vision)    │
│  │       writes ──► suggestions/*.json                               │
│  │                                                                   │
│  ├── scripts/update_tags.py                                          │
│  │       reads  ◄── suggestions/*.json                               │
│  │       writes ──► data/tags.json                                   │
│  │                                                                   │
│  └── simulator/index.html  (opens in browser, no server needed)     │
└─────────────────────────────────────────────────────────────────────┘
          │                              │
          ▼                              ▼
   n8n (artworks)               mod.artworks.ai
   triggers Piper                shows results,
   pipeline re-run               Export → JSON
          │
          ▼
   Piper pipeline
   (violations-detector-test-v3)
   ┌───────────────┐
   │ prepare_params│
   │ siglip_config │
   │ ask_siglip2   │ ← uses data/tags.json (via Piper UI)
   │ evaluate_siglip│
   └───────────────┘
```

## Data flow for one improvement cycle

```
1. run_category.py
   POST /webhook → n8n → Piper pipeline runs on all test images in category
   
2. mod.artworks.ai
   Human: Refresh → Export → JSON saved to EXPORT_WATCH_DIR
   
3. analyze_failed.py
   Load JSON → filter Failed+ and Failed- items
   For each failed item:
     GET image_url
     POST to OpenRouter (Grok-4.1-fast) with image + context prompt
     Parse JSON response with tag suggestions
   Save aggregated suggestions → suggestions/<category>_<date>.json
   
4. update_tags.py
   Load suggestions → show diff table (add / improve / remove)
   Human confirms each change (or --auto flag)
   Write updated data/tags.json
   Save changelog → suggestions/<category>_<date>_applied.json
   
5. Go to step 1 (verify improvements)
```

## Key design decisions

### Why no server for the simulator?
The simulator is a single-page React app transpiled by Babel in the browser. 
This means zero setup — open the HTML file and it works. CDN dependencies 
require internet access.

### Why Python for scripts, not Node.js?
Python has a richer ecosystem for data processing, better CLI libraries (click, rich),
and is more familiar to data/ML teams who might extend this project.

### Why aggregate suggestions before asking for confirmation?
Multiple failed images often suggest the same tag improvements. By aggregating 
first, you see a "votes" count — 5 images suggesting the same new tag is much 
more convincing than 1. This reduces noise and focuses human review.

### Why keep tags.json in git?
Tags are the core "model parameters" of the SigLIP-2 system. Versioning them 
means you can always roll back to a known-good state and see the history of 
improvements over time.
