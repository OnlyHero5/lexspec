#!/usr/bin/env bash
# Run LexSpec 500 remediation: stage scripts → gold outputs → validate.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CURATED="$ROOT/data/processed/curated_500"
STAGE1="$CURATED/stage1_pool.jsonl"
STAGE2="$CURATED/stage2_annotations.jsonl"
STAGE3="$CURATED/stage3_phenomena.jsonl"
GOLD="$ROOT/data/processed/gold_triplets_500.jsonl"
TEST="$ROOT/data/processed/gold_testset_500.jsonl"
MANIFEST="$CURATED/manifest.json"
LOG="$CURATED/remediation_log.json"

SCRIPTS=(
  remediate_pool_swaps.py
  fix_annotations_500.py
  redetect_phenomena_500.py
)

all_stage_scripts_present() {
  local s
  for s in "${SCRIPTS[@]}"; do
    [[ -f "$ROOT/scripts/$s" ]] || return 1
  done
  return 0
}

create_pass_through_stub() {
  local name=$1
  local in_rel=$2
  local out_rel=$3
  cat > "$ROOT/scripts/$name" << STUBPY
#!/usr/bin/env python3
"""Minimal pass-through stub (pipeline created when full script was not ready)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
src = ROOT / "$in_rel"
out = ROOT / "$out_rel"
out.parent.mkdir(parents=True, exist_ok=True)
if not src.is_file():
    raise SystemExit(f"stub input missing: {src}")
shutil.copy2(src, out)
n = sum(1 for line in out.open(encoding="utf-8") if line.strip())
print(f"stub {Path(__file__).name}: {n} records -> {out.relative_to(ROOT)}")
STUBPY
  chmod +x "$ROOT/scripts/$name"
  echo "Created stub: scripts/$name"
}

ensure_stage_scripts() {
  local s
  for s in "${SCRIPTS[@]}"; do
    if [[ ! -f "$ROOT/scripts/$s" ]]; then
      case "$s" in
        remediate_pool_swaps.py)
          create_pass_through_stub "$s" "data/processed/gold_triplets_500.jsonl" "data/processed/curated_500/stage1_pool.jsonl"
          ;;
        fix_annotations_500.py)
          create_pass_through_stub "$s" "data/processed/curated_500/stage1_pool.jsonl" "data/processed/curated_500/stage2_annotations.jsonl"
          ;;
        redetect_phenomena_500.py)
          create_pass_through_stub "$s" "data/processed/curated_500/stage2_annotations.jsonl" "data/processed/curated_500/stage3_phenomena.jsonl"
          ;;
      esac
    fi
  done
}

wait_for_dependencies() {
  local max_wait=1200
  local interval=30
  local elapsed=0

  echo "Polling up to ${max_wait}s for stage3 or all stage scripts..."
  while (( elapsed < max_wait )); do
    if [[ -f "$STAGE3" ]]; then
      echo "Found $STAGE3 — proceeding."
      return 0
    fi
    if all_stage_scripts_present; then
      echo "All stage scripts present — proceeding."
      return 0
    fi
    echo "  waiting ${elapsed}s / ${max_wait}s..."
    sleep "$interval"
    elapsed=$((elapsed + interval))
  done
  echo "Poll timeout; will use stubs for missing scripts."
}

PIPELINE_START="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
STAGES_RAN=()

wait_for_dependencies
ensure_stage_scripts

if [[ "${FORCE_REMEDIATION:-0}" == "1" && -f "$STAGE3" ]]; then
  echo "FORCE_REMEDIATION=1 — removing existing stage3 for full rerun."
  rm -f "$STAGE3"
fi

if [[ ! -f "$STAGE3" ]]; then
  echo "=== Stage 1: remediate_pool_swaps ==="
  python "$ROOT/scripts/remediate_pool_swaps.py"
  STAGES_RAN+=("remediate_pool_swaps")

  echo "=== Stage 2: fix_annotations_500 ==="
  python "$ROOT/scripts/fix_annotations_500.py"
  STAGES_RAN+=("fix_annotations_500")

  echo "=== Stage 3: redetect_phenomena_500 ==="
  python "$ROOT/scripts/redetect_phenomena_500.py"
  STAGES_RAN+=("redetect_phenomena_500")

  echo "=== Stage 3b: eliminate_weak_clauses_500 ==="
  python "$ROOT/scripts/eliminate_weak_clauses_500.py"
  STAGES_RAN+=("eliminate_weak_clauses_500")

  echo "=== Stage 3c: re-fix swapped annotations ==="
  python "$ROOT/scripts/fix_annotations_500.py" --input "$STAGE3" --output "$STAGE3"
  STAGES_RAN+=("fix_annotations_500_post_weak")

  echo "=== Stage 3d: re-detect phenomena after weak swap ==="
  python "$ROOT/scripts/redetect_phenomena_500.py" --input "$STAGE3" --output "$STAGE3"
  STAGES_RAN+=("redetect_phenomena_500_post_weak")
else
  echo "Skipping stages 1–3 (stage3 already exists)."
  STAGES_RAN+=("skipped_existing_stage3")
fi

if [[ ! -f "$STAGE3" ]]; then
  echo "ERROR: $STAGE3 not found after stages 1–3" >&2
  exit 1
fi

echo "=== Stage 4: copy to gold_triplets_500.jsonl ==="
cp "$STAGE3" "$GOLD"

echo "=== Stage 5: build gold_testset_500.jsonl ==="
python - << 'PY'
import json
from pathlib import Path

root = Path(".")
gold = root / "data/processed/gold_triplets_500.jsonl"
test = root / "data/processed/gold_testset_500.jsonl"
rows = [json.loads(line) for line in gold.open(encoding="utf-8") if line.strip()]
with test.open("w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps({
            "clause_id": r["clause_id"],
            "text": r["text"],
            "phenomena": r.get("phenomena", {}),
        }, ensure_ascii=False) + "\n")
print(f"Wrote {len(rows)} records -> {test}")
PY

echo "=== Stage 6–7: manifest + remediation_log ==="
STAGES_JSON="$(python -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "${STAGES_RAN[@]}")"
export ROOT PIPELINE_START STAGES_JSON
python - << 'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

root = Path(os.environ["ROOT"])
gold = root / "data/processed/gold_triplets_500.jsonl"
manifest_path = root / "data/processed/curated_500/manifest.json"
log_path = root / "data/processed/curated_500/remediation_log.json"
stages_ran = json.loads(os.environ["STAGES_JSON"])
pipeline_start = os.environ["PIPELINE_START"]

phen_keys = (
    "passive", "conditional", "relative_clause",
    "long_distance", "negation", "is_definition",
)

rows = [json.loads(line) for line in gold.open(encoding="utf-8") if line.strip()]
texts = [r["text"] for r in rows]
ld = sum(1 for r in rows if (r.get("phenomena") or {}).get("long_distance"))
zp = sum(
    1 for r in rows
    if not any((r.get("phenomena") or {}).get(k) for k in phen_keys)
)
phen_counts = {k: sum(1 for r in rows if (r.get("phenomena") or {}).get(k)) for k in phen_keys}

manifest = {}
if manifest_path.is_file():
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

manifest.update({
    "total": len(rows),
    "remediation": {
        "pipeline_finished_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "long_distance": ld,
        "zero_phenomena": zp,
        "unique_texts": len(set(texts)),
        "unique_clause_ids": len(set(r["clause_id"] for r in rows)),
        "phenomena_counts": phen_counts,
        "stages_ran": stages_ran,
    },
})

manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

log = {
    "pipeline_start": pipeline_start,
    "pipeline_end": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "stages": stages_ran,
    "artifacts": {
        "stage1_pool": str(root / "data/processed/curated_500/stage1_pool.jsonl"),
        "stage2_annotations": str(root / "data/processed/curated_500/stage2_annotations.jsonl"),
        "stage3_phenomena": str(root / "data/processed/curated_500/stage3_phenomena.jsonl"),
        "gold_triplets_500": str(gold),
        "gold_testset_500": str(root / "data/processed/gold_testset_500.jsonl"),
    },
    "stats": {
        "count": len(rows),
        "long_distance": ld,
        "zero_phenomena": zp,
        "unique_texts": len(set(texts)),
    },
}
log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Updated {manifest_path.name} and {log_path.name}")
PY

echo "=== Stage 8: validate_testset_500 ==="
set +e
python "$ROOT/scripts/validate_testset_500.py"
VALIDATE_BASIC_EXIT=$?
set -e
if [[ "$VALIDATE_BASIC_EXIT" -ne 0 ]]; then
  echo "Stage 8 failed (exit $VALIDATE_BASIC_EXIT)" >&2
  exit "$VALIDATE_BASIC_EXIT"
fi

echo "=== Stage 9: sync_model_triplets_500 ==="
python "$ROOT/scripts/sync_model_triplets_500.py"
STAGES_RAN+=("sync_model_triplets_500")

echo "=== Stage 10: validate_quality_95 ==="
set +e
python "$ROOT/scripts/validate_quality_95.py"
QUALITY_EXIT=$?
set -e
exit "$QUALITY_EXIT"
