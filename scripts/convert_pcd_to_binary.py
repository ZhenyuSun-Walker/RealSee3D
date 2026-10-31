"""Convert existing scene point clouds to verified binary PLY, atomically."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time
import traceback

try:
    from .ply_io import convert_to_binary, read_header, validate_binary
except ImportError:
    from ply_io import convert_to_binary, read_header, validate_binary


def pointcloud_files(data_root):
    for group in ("real_world_data", "synthetic_data"):
        folder = data_root / group
        if not folder.is_dir():
            continue
        for scene in sorted(folder.iterdir()):
            if scene.is_dir():
                yield from sorted((scene / "pointclouds").glob("*.ply"))


def audit(data_root):
    result = {"checked_at": time.time(), "files": 0, "bytes": 0,
              "formats": Counter(), "record_bytes": Counter(), "errors": []}
    for path in pointcloud_files(data_root):
        result["files"] += 1
        try:
            with path.open("rb") as stream:
                header = read_header(stream)
            result["formats"][header.format] += 1
            if header.format != "binary_little_endian":
                result["errors"].append({"path": str(path), "error": "not binary_little_endian"})
                continue
            header = validate_binary(path)
            result["record_bytes"][header.dtype.itemsize] += 1
            result["bytes"] += path.stat().st_size
        except Exception as error:
            result["errors"].append({"path": str(path), "error": str(error)})
    result["formats"] = dict(result["formats"])
    result["record_bytes"] = dict(result["record_bytes"])
    return result


def convert_one(path, chunk_vertices):
    started = time.monotonic()
    result = convert_to_binary(path, chunk_vertices)
    result["elapsed_seconds"] = time.monotonic() - started
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path,
                        default=Path(__file__).resolve().parent.parent / "data/RealSee3D")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--chunk-vertices", type=int, default=250000)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Append per-file conversion and verification records as JSONL")
    parser.add_argument("--include-incomplete", action="store_true",
                        help="Also inspect scenes without .complete; only use after old writers stop")
    parser.add_argument("--audit-only", action="store_true",
                        help="Audit every current PLY header and binary payload size without conversion")
    args = parser.parse_args()
    if args.workers < 1 or args.chunk_vertices < 1:
        parser.error("Worker and chunk counts must be positive")
    if not args.data_root.is_dir():
        parser.error("Dataset root does not exist")
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    if args.audit_only:
        result = audit(args.data_root)
        args.manifest.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        return int(bool(result["errors"]))

    paths = []
    deferred = []
    for path in pointcloud_files(args.data_root):
        if args.include_incomplete or (path.parent / ".complete").is_file():
            paths.append(path)
        else:
            deferred.append(str(path))
    summary = {"started_at": time.time(), "selected_files": len(paths),
               "deferred": deferred, "converted": 0, "already_binary": 0,
               "before_bytes": 0, "after_bytes": 0, "failures": []}
    print(f"Selected {len(paths)} PLY files; deferred {len(deferred)} incomplete scenes; "
          f"workers={args.workers}", flush=True)
    with args.manifest.open("a", buffering=1) as manifest:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(convert_one, path, args.chunk_vertices): path for path in paths}
            for index, future in enumerate(as_completed(futures), 1):
                path = futures[future]
                try:
                    result = future.result()
                    summary[result["action"]] += 1
                    summary["before_bytes"] += result["before_bytes"]
                    summary["after_bytes"] += result["after_bytes"]
                    manifest.write(json.dumps(result) + "\n")
                    print(f"[{index}/{len(paths)}] {result['action']}: {path} "
                          f"{result['before_bytes']} -> {result['after_bytes']} bytes "
                          f"({result['elapsed_seconds']:.2f}s)", flush=True)
                except Exception as error:
                    failure = {"path": str(path), "action": "failed", "error": str(error)}
                    summary["failures"].append(failure)
                    manifest.write(json.dumps(failure) + "\n")
                    print(f"FAILED: {path}\n{traceback.format_exc()}", flush=True)
    summary["finished_at"] = time.time()
    summary["saved_bytes"] = summary["before_bytes"] - summary["after_bytes"]
    summary_path = args.manifest.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print("SUMMARY: " + json.dumps(summary), flush=True)
    return int(bool(summary["failures"]))


if __name__ == "__main__":
    raise SystemExit(main())
