#!/usr/bin/env bash

set -u
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-/home/lf/data/thermal3dgs/RGBT-Scenes}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/output}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-thermal3dgs_ommg}"
ANCHOR_STATS_EMA="${ANCHOR_STATS_EMA:-0.95}"

run_python() {
  if [[ "${CONDA_DEFAULT_ENV:-}" == "$CONDA_ENV_NAME" ]]; then
    python "$@"
  else
    conda run -n "$CONDA_ENV_NAME" python "$@"
  fi
}

collect_scenes() {
  if [[ $# -gt 0 ]]; then
    local requested_scene
    for requested_scene in "$@"; do
      if [[ -d "$requested_scene" ]]; then
        printf '%s\n' "$requested_scene"
      elif [[ -d "$DATA_ROOT/$requested_scene" ]]; then
        printf '%s\n' "$DATA_ROOT/$requested_scene"
      else
        printf '[Warning] Skip unknown scene: %s\n' "$requested_scene" >&2
      fi
    done
    return
  fi

  find "$DATA_ROOT" -mindepth 1 -maxdepth 1 -type d | sort
}

main() {
  local -a scenes=()
  local scene_dir=""
  local scene_name=""
  local out_dir=""
  local -a failures=()

  if [[ ! -d "$DATA_ROOT" ]]; then
    printf '[Error] DATA_ROOT does not exist: %s\n' "$DATA_ROOT" >&2
    exit 1
  fi

  mapfile -t scenes < <(collect_scenes "$@")
  if [[ ${#scenes[@]} -eq 0 ]]; then
    printf '[Error] No scene directories found under %s\n' "$DATA_ROOT" >&2
    exit 1
  fi

  mkdir -p "$OUTPUT_ROOT"

  printf 'Repo: %s\n' "$ROOT_DIR"
  printf 'Data root: %s\n' "$DATA_ROOT"
  printf 'Output root: %s\n' "$OUTPUT_ROOT"
  printf 'Conda env: %s\n' "$CONDA_ENV_NAME"
  printf 'Scenes: %s\n' "${#scenes[@]}"

  for scene_dir in "${scenes[@]}"; do
    scene_name="$(basename "$scene_dir")"
    out_dir="$OUTPUT_ROOT/$scene_name"

    printf '\n===== [%s] train =====\n' "$scene_name"
    if ! run_python "$ROOT_DIR/train-OMMG.py" \
      -s "$scene_dir" \
      -m "$out_dir" \
      --use_paired_views \
      --anchor_stats_ema "$ANCHOR_STATS_EMA" \
      --save_anchor_stats; then
      failures+=("$scene_name:train")
      continue
    fi

    printf '===== [%s] render =====\n' "$scene_name"
    if ! run_python "$ROOT_DIR/render.py" -m "$out_dir"; then
      failures+=("$scene_name:render")
      continue
    fi

    printf '===== [%s] metrics =====\n' "$scene_name"
    if ! run_python "$ROOT_DIR/metrics.py" -m "$out_dir"; then
      failures+=("$scene_name:metrics")
      continue
    fi
  done

  printf '\n===== Summary =====\n'
  if [[ ${#failures[@]} -eq 0 ]]; then
    printf 'All scenes completed successfully.\n'
    exit 0
  fi

  printf 'Failed stages:\n'
  printf '  %s\n' "${failures[@]}"
  exit 1
}

main "$@"
