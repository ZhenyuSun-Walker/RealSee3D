#!/usr/bin/env bash
# Generate point clouds for extracted real-world and synthetic scenes.
# Activate the Conda environment containing numpy and cv2 before running.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
data_root="$repo_root/data/RealSee3D"
python_bin="${PYTHON_BIN:-python}"
save_individual=true
force=false

usage() {
    cat <<'EOF'
Usage: bash scripts/generate_all_pcd.sh [options]

Options:
  --data-root PATH  Dataset root containing real_world_data/ and synthetic_data/.
  --merged-only     Save only merged.ply; omit individual viewpoint point clouds.
  --force           Reprocess scenes even if their .complete marker exists.
  -h, --help        Show this help.

Outputs: <scene>/pointclouds/{merged.ply,<viewpoint_id>.ply,generate.log,.complete}
Activate your Conda environment first. PYTHON_BIN can override the interpreter.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-root)
            if [[ $# -lt 2 || -z "$2" ]]; then
                echo "Error: --data-root requires a path." >&2
                exit 2
            fi
            data_root="$2"
            shift 2
            ;;
        --merged-only) save_individual=false; shift ;;
        --force) force=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Error: unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ ! -d "$data_root" ]]; then
    echo "Error: dataset directory not found: $data_root" >&2
    exit 1
fi

if ! "$python_bin" -c 'import numpy, cv2'; then
    echo "Error: activate a Python environment containing numpy and cv2." >&2
    exit 1
fi

extra_args=()
if [[ "$save_individual" == true ]]; then
    extra_args+=(--save_individual)
fi

completed=0
skipped=0
failed=0
shopt -s nullglob

for scene in "$data_root"/real_world_data/* "$data_root"/synthetic_data/*; do
    [[ -d "$scene/viewpoints" ]] || continue
    out="$scene/pointclouds"
    mkdir -p -- "$out"

    if [[ "$force" == false && -f "$out/.complete" ]]; then
        echo "Skipping completed scene: $scene"
        skipped=$((skipped + 1))
        continue
    fi

    # Remove the old marker before a forced run, so interrupted runs can retry.
    rm -f -- "$out/.complete"
    echo "Processing: $scene"
    if "$python_bin" -u "$script_dir/generate_pcd.py" \
        --source "$scene" \
        --output "$out" \
        "${extra_args[@]}" > "$out/generate.log" 2>&1 \
        && [[ -s "$out/merged.ply" ]] \
        && tail -n 1 -- "$out/generate.log" | grep -Eq '^Done\. Saved [1-9][0-9]* points\.$'; then
        touch -- "$out/.complete"
        completed=$((completed + 1))
        echo "Completed: $scene"
    else
        failed=$((failed + 1))
        echo "Failed: $scene (see $out/generate.log)" >&2
    fi
done

echo "Finished: completed=$completed skipped=$skipped failed=$failed"
if [[ $((completed + skipped + failed)) -eq 0 ]]; then
    echo "Error: no extracted scenes found in real_world_data/ or synthetic_data/." >&2
    exit 1
fi
if [[ "$failed" -gt 0 ]]; then
    exit 1
fi
