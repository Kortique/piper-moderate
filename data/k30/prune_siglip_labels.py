"""Modify build_new_project.py output: shrink the ask_siglip2.labels default
to only include the underage/adult labels our pruned LGBM actually uses.

Keep all NON-underage/adult labels (bestiality, blood, death, weapons, drugs,
rape, asian/ebony/arab/latin/indian, human_waste, no_*) untouched — those feed
the other piper nodes (lgbm_bestiality, evaluate_siglip) which we don't own.

This is where the SigLIP latency speedup actually comes from.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent

UNDERAGE_PREFIX = "underage_"
ADULT_PREFIX = "adult_"


def main():
    payload = json.loads((ROOT / "piper_new_project_payload.json").read_text())
    keepers = json.loads((ROOT / "feature_keepers.json").read_text())
    keep_siglip = set(keepers["siglip_keepers"])  # piper feature names, e.g. "little_girl" or "adult__woman"

    # Convert to label keys the way ask_siglip2.labels expects:
    #   feature "little_girl" → label key "underage_little_girl"
    #   feature "underage_X:x20" was stripped to "X"; but the original key may have :xN suffix
    # We need to keep ALL variants of the feature name in the label map.
    keep_label_keys = set()
    for f in keep_siglip:
        if f.startswith("adult__"):
            keep_label_keys.add(ADULT_PREFIX + f[len("adult__"):])
        else:
            keep_label_keys.add(UNDERAGE_PREFIX + f)

    # Read existing labels dict
    nodes = payload["pipeline"]["nodes"]
    askl = nodes["ask_siglip2"]
    labels_default = askl["inputs"]["labels"]["default"]
    if isinstance(labels_default, str):
        labels_dict = json.loads(labels_default)
    else:
        labels_dict = labels_default

    # CSAM-ONLY MODE: drop all non-underage/non-adult categories entirely.
    # Bestiality, weapons, blood, death, drugs, rape, ethnicity, human_waste, no_*
    # labels feed OTHER piper nodes (lgbm_bestiality, evaluate_siglip's other
    # categories) — those nodes will still run but produce zeros for unrelated
    # categories. This pipeline becomes CSAM-only.
    new_labels = {}
    n_under_dropped = 0
    n_under_kept = 0
    n_adult_dropped = 0
    n_adult_kept = 0
    n_other_dropped = 0
    for k, v in labels_dict.items():
        prefix = None
        if k.startswith(UNDERAGE_PREFIX):
            prefix = UNDERAGE_PREFIX
        elif k.startswith(ADULT_PREFIX):
            prefix = ADULT_PREFIX
        if prefix is None:
            # different category — DROP for CSAM-only minimal pipeline
            n_other_dropped += 1
            continue
        # strip :xN suffix to compare
        m = re.search(r":x(\d+(?:\.\d+)?)$", k)
        base_key = k[:m.start()] if m else k
        if base_key in keep_label_keys:
            new_labels[k] = v
            if prefix == UNDERAGE_PREFIX: n_under_kept += 1
            else: n_adult_kept += 1
        else:
            if prefix == UNDERAGE_PREFIX: n_under_dropped += 1
            else: n_adult_dropped += 1

    askl["inputs"]["labels"]["default"] = json.dumps(new_labels)
    (ROOT / "piper_new_project_payload.json").write_text(json.dumps(payload))

    print(f"underage labels: kept {n_under_kept}, dropped {n_under_dropped}")
    print(f"adult labels:    kept {n_adult_kept}, dropped {n_adult_dropped}")
    print(f"OTHER categories dropped (CSAM-only mode): {n_other_dropped}")
    print(f"total labels in ask_siglip2.labels: {len(labels_dict)} → {len(new_labels)} "
          f"({len(labels_dict)-len(new_labels)} dropped = {(len(labels_dict)-len(new_labels))/len(labels_dict):.0%})")


if __name__ == "__main__":
    main()
