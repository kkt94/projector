#!/bin/bash
# Run PLM + VL3, vanilla + 5 ckpts each, on FULL MVBench (max_per_task=0).
# N concurrent jobs per GPU (N_PLM, N_VL3) to soak up VRAM.

set -u
cd "$(dirname "$0")/.."

export HF_HOME=/home/dev/projects/projector/.cache/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_MODULES_CACHE="$HF_HOME/modules"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PATH="/home/dev/projects/projector/.venv/bin:$PATH"
mkdir -p "$HF_MODULES_CACHE"

LOG_DIR=tools/mvbench_full_logs
RES_DIR=outputs/mvbench_full
mkdir -p "$LOG_DIR" "$RES_DIR"
log() { echo "[$(date +%F\ %T)] $*"; }

PLM_DIR=outputs/plm3b_run01
VL3_DIR=outputs/vl3_7b_run01
PLM_CKPTS=(ckpt_step_200 ckpt_step_400 ckpt_step_600 ckpt_step_800)
VL3_CKPTS=(ckpt_step_200 ckpt_step_400 ckpt_step_600 ckpt_step_800 ckpt_step_1000)

# Concurrent jobs per GPU per family.
N_PLM="${N_PLM:-3}"   # PLM-3B ~9GB/job → 3×9=27GB per GPU
N_VL3="${N_VL3:-2}"   # VL3-7B ~25GB/job → 2×25=50GB per GPU

run_one() {
    local gpu="$1"; local script="$2"; local variant="$3"
    local ckpt="$4"; local out="$5"; local lg="$6"; local base="$7"
    local args=(--variant "$variant" --gpu "$gpu" --output "$out"
                --base_model "$base" --max_per_task 0)
    [ "$variant" != "vanilla" ] && args+=(--checkpoint_dir "$ckpt")
    if [ -f "$out" ]; then
        log "  skip (exists): $out"; return 0
    fi
    log "[gpu$gpu] $(basename $script) $variant ${ckpt##*/}"
    .venv/bin/python "$script" "${args[@]}" > "$lg" 2>&1
    local rc=$?
    [ "$rc" -ne 0 ] && log "  FAILED rc=$rc -> $lg"
    return $rc
}

# Process one family with N concurrent slots per GPU.
# args: tag root script base N ckpt1 ckpt2 ...
run_family() {
    local tag="$1"; local root="$2"; local script="$3"; local base="$4"; local N="$5"
    shift 5
    local ckpts=("$@")

    # Build job specs: "variant|ckpt|out|lg"
    local jobs=()
    jobs+=("vanilla||$RES_DIR/${tag}_mvb_full_vanilla.json|$LOG_DIR/${tag}_vanilla.log")
    for c in "${ckpts[@]}"; do
        jobs+=("phase_a|$root/$c|$RES_DIR/${tag}_mvb_full_${c}.json|$LOG_DIR/${tag}_${c}.log")
    done

    log "[$tag] queued ${#jobs[@]} jobs, N=$N per GPU (so $((2*N)) concurrent total)"

    # Slot tracking: parallel arrays of PIDs per GPU.
    declare -a pids0=() pids1=()
    local gpu=0
    for spec in "${jobs[@]}"; do
        IFS='|' read -r v c o l <<< "$spec"

        # Wait for a slot on the chosen gpu if full.
        while true; do
            if [ "$gpu" = "0" ]; then
                local active=()
                for p in "${pids0[@]:-}"; do kill -0 "$p" 2>/dev/null && active+=("$p"); done
                pids0=("${active[@]}")
                [ "${#pids0[@]}" -lt "$N" ] && break
            else
                local active=()
                for p in "${pids1[@]:-}"; do kill -0 "$p" 2>/dev/null && active+=("$p"); done
                pids1=("${active[@]}")
                [ "${#pids1[@]}" -lt "$N" ] && break
            fi
            sleep 5
        done

        run_one "$gpu" "$script" "$v" "$c" "$o" "$l" "$base" &
        local pid=$!
        if [ "$gpu" = "0" ]; then pids0+=("$pid"); else pids1+=("$pid"); fi

        # Round-robin GPU.
        gpu=$((1 - gpu))
    done

    # Wait for all remaining.
    for p in "${pids0[@]:-}" "${pids1[@]:-}"; do
        wait "$p" 2>/dev/null || true
    done
    log "[$tag] all jobs done"
}

# Phase 1: PLM (smaller, more concurrent). 4 ckpts (200/400/600/800).
run_family plm "$PLM_DIR" scripts/eval_mvbench_plm3b.py facebook/Perception-LM-3B "$N_PLM" "${PLM_CKPTS[@]}"

# Phase 2: VL3 (larger). 5 ckpts (200/400/600/800/1000).
run_family vl3 "$VL3_DIR" scripts/eval_mvbench_vl3.py models/VideoLLaMA3-7B "$N_VL3" "${VL3_CKPTS[@]}"

log "ALL DONE"
echo
echo "=== Summary ==="
for j in "$RES_DIR"/*.json; do
    python3 -c "
import json
try:
    d=json.load(open('$j'))
    name='$(basename $j .json)'
    print(f'{name}: overall={d[\"overall_accuracy\"]:.3f}  total={d[\"total_correct\"]}/{d[\"total_eval\"]}')
except Exception as e:
    print(f'$(basename $j .json): ERR {e}')
" 2>/dev/null
done
