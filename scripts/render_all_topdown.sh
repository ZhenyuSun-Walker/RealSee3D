#!/usr/bin/env bash
# Render existing merged point clouds sequentially with the active Python env.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
data_root="$repo_root/data/RealSee3D"
python_bin="${PYTHON_BIN:-python}"
subset=all
force=false
producer_pid=""
poll_seconds=30
render_args=()

usage() {
    cat <<'EOF'
Usage: bash scripts/render_all_topdown.sh [options]

  --data-root PATH          Root containing real_world_data/ and synthetic_data/.
  --subset all|real|synthetic  Which scenes to process (default: all).
  --force                   Replace existing rgb_td.png files.
  --follow-pcd-pid PID      Follow an active generate_all_pcd.sh job; wait for each
                            pointclouds/.complete marker before rendering.
  --poll-seconds NUMBER     Delay between checks for newly completed clouds (default: 30).
  --height-ratio NUMBER     Floor-to-ceiling height fraction above floor (default: 0.45).
  --cut-height NUMBER       Optional absolute cut in meters instead of a height ratio.
  --y-min NUMBER            Override the automatic world-Y cut minimum.
  --y-max NUMBER            Override the automatic world-Y floor maximum.
  --output-size NUMBER      Square image size in pixels (default: 4096); 0 uses metric resolution.
  --resolution NUMBER       Meters per pixel with --output-size 0 (default: 0.005).
  --splat-radius NUMBER     Point coverage radius in meters (default: 0.02; diameter 0.04).
  --color-mode blend|original  Height-gated RGB fusion or original point colors (default: blend).
  --blend-height NUMBER     Same-surface height tolerance in meters (default: 0.04).
  --trim-percent NUMBER     Horizontal tail trimming (default: 0.1).
  --fill-pixels NUMBER      Small-hole fill distance (default: 2).
  --no-cleanup              Disable main-footprint cleanup.
  --cleanup-radius NUMBER   Cleanup neighborhood radius in meters (default: 0.15).
  --cleanup-density NUMBER  Minimum seed occupancy (default: 0.1).
  -h, --help                Show help.

Input:  <scene>/pointclouds/merged.ply
Output: <scene>/rgb_td/rgb_td.png, with per-scene render.log.
Scenes missing merged.ply are skipped. Existing nonempty PNGs are skipped unless
--force is used. Scenes with multiple floor IDs are skipped. Floor and ceiling
planes are estimated for each single-floor scene; only points below the cut
are rendered. The default cut is 45% of each estimated floor-to-ceiling height,
measured upward from the floor. --height-ratio values are dimensionless. Exterior
isolated samples are removed by main-footprint cleanup, enabled by default.
RGB blending uses Gaussian weights with soft splat edges, restricted to a local
footprint and the visible surface's height band. --color-mode original disables blending. Square output preserves aspect
ratio and adds padding; its metric scale adapts per scene.
Activate a Conda environment containing numpy and cv2 first.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-root|--subset|--follow-pcd-pid|--poll-seconds|--cut-height|--height-ratio|--y-min|--y-max|--output-size|--resolution|--splat-radius|--color-mode|--blend-height|--trim-percent|--fill-pixels|--cleanup-radius|--cleanup-density)
            if [[ $# -lt 2 || -z "$2" ]]; then
                echo "Error: $1 requires a value." >&2
                exit 2
            fi
            case "$1" in
                --data-root) data_root="$2" ;;
                --subset) subset="$2" ;;
                --follow-pcd-pid) producer_pid="$2" ;;
                --poll-seconds) poll_seconds="$2" ;;
                *) render_args+=("$1" "$2") ;;
            esac
            shift 2
            ;;
        --force) force=true; shift ;;
        --no-cleanup) render_args+=(--no-cleanup); shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Error: unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -n "$producer_pid" && ! "$producer_pid" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: --follow-pcd-pid requires a positive process ID." >&2
    exit 2
fi
if [[ ! "$poll_seconds" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: --poll-seconds requires a positive integer." >&2
    exit 2
fi

case "$subset" in
    all) subsets=(real_world_data synthetic_data) ;;
    real) subsets=(real_world_data) ;;
    synthetic) subsets=(synthetic_data) ;;
    *) echo "Error: --subset must be all, real, or synthetic." >&2; exit 2 ;;
esac
[[ -d "$data_root" ]] || { echo "Error: missing dataset: $data_root" >&2; exit 1; }
if ! "$python_bin" -c 'import numpy, cv2'; then
    echo "Error: activate a Python environment containing numpy and cv2." >&2
    exit 1
fi

completed=0
skipped=0
missing=0
failed=0
multi_floor=0
found=0
pending=""
trap 'if [[ -n "$pending" ]]; then rm -f -- "$pending"; fi' EXIT
shopt -s nullglob

scene_queue=()
for group in "${subsets[@]}"; do
    for scene in "$data_root/$group"/*; do
        [[ -d "$scene/viewpoints" ]] || continue
        found=$((found + 1))
        scene_queue+=("$scene")
    done
done

while [[ ${#scene_queue[@]} -gt 0 ]]; do
    waiting_scenes=()
    for scene in "${scene_queue[@]}"; do
        source_ply="$scene/pointclouds/merged.ply"
        out="$scene/rgb_td"
        if [[ "$force" == false && -s "$out/rgb_td.png" ]]; then
            skipped=$((skipped + 1))
            echo "Skipping existing image: $out/rgb_td.png"
            continue
        fi
        if [[ -n "$producer_pid" && ! -f "$scene/pointclouds/.complete" ]] \
            && kill -0 "$producer_pid" 2>/dev/null; then
            waiting_scenes+=("$scene")
            continue
        fi
        if [[ ! -s "$source_ply" ]]; then
            missing=$((missing + 1))
            echo "Skipping missing point cloud: $source_ply"
            continue
        fi
        mkdir -p -- "$out"
        pending="$out/.rgb_td.pending.$$.png"
        echo "Rendering: $scene"
        if "$python_bin" -u "$script_dir/render_topdown.py" \
            --source "$source_ply" --output "$pending" \
            "${render_args[@]}" > "$out/render.log" 2>&1 \
            && [[ -s "$pending" ]]; then
            mv -f -- "$pending" "$out/rgb_td.png"
            completed=$((completed + 1))
            echo "Saved: $out/rgb_td.png"
        else
            status=$?
            rm -f -- "$pending"
            if [[ "$status" -eq 3 ]]; then
                multi_floor=$((multi_floor + 1))
                echo "Skipping multi-floor scene: $scene"
            else
                failed=$((failed + 1))
                echo "Failed: $scene (see $out/render.log)" >&2
            fi
        fi
        pending=""
    done
    echo "Progress: completed=$completed existing=$skipped waiting_pcd=${#waiting_scenes[@]} missing_ply=$missing multi_floor=$multi_floor failed=$failed"
    scene_queue=("${waiting_scenes[@]}")
    if [[ ${#scene_queue[@]} -gt 0 ]] && kill -0 "$producer_pid" 2>/dev/null; then
        echo "Waiting ${poll_seconds}s for point-cloud producer PID $producer_pid..."
        sleep "$poll_seconds"
    fi
done

echo "Finished: completed=$completed existing=$skipped missing_ply=$missing multi_floor=$multi_floor failed=$failed"
if [[ "$found" -eq 0 || $((completed + skipped)) -eq 0 || "$failed" -gt 0 ]]; then
    exit 1
fi
