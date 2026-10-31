"""Resume failed perspective scenes, optionally following an active batch log."""

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time

from export_perspectives import run_scene, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-log", type=Path, required=True)
    parser.add_argument("--follow-pid", type=int, help="Follow this original exporter until it exits.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--poll-seconds", type=float, default=30.)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    if args.workers < 1 or not 0 < args.poll_seconds <= 60:
        parser.error("workers must be positive; poll-seconds must be in (0,60]")
    if args.follow_pid is not None and args.follow_pid <= 0:
        parser.error("follow-pid must be positive")
    with args.batch_log.open() as stream:
        header = json.loads(stream.readline())
    config = header["config"]
    results, submitted, pending = {}, set(), {}
    halted = None
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        while True:
            failures = {}
            batch_text = args.batch_log.read_text()
            for line in batch_text.splitlines():
                try:
                    record = json.loads(line)
                except ValueError:
                    continue  # A concurrently written final line will be read next time.
                if record.get("status") == "failed":
                    failures[record["scene"]] = record
            for scene in failures:
                if scene in submitted or len(pending) >= args.workers or halted:
                    continue
                if shutil.disk_usage(scene).free < 50 * 1024 ** 3:
                    halted = "Free disk space below 50 GiB"
                    break
                submitted.add(scene)
                pending[pool.submit(run_scene, scene, config, False)] = scene
            if pending:
                done, _ = wait(pending, timeout=args.poll_seconds, return_when=FIRST_COMPLETED)
                for future in done:
                    scene = pending.pop(future)
                    try:
                        result = future.result()
                    except Exception as error:
                        result = {"scene": scene, "status": "failed", "error": str(error)}
                        halted = "A repair worker failed unexpectedly"
                    results[scene] = result
                    print(json.dumps(result), flush=True)
            producer_alive = args.follow_pid is not None and Path(f"/proc/{args.follow_pid}").exists()
            report = {"updated_utc": datetime.now(timezone.utc).isoformat(),
                      "batch_log": str(args.batch_log), "producer_alive": producer_alive,
                      "failed_scenes_seen": len(failures),
                      "repaired": sum(r["status"] == "completed" for r in results.values()),
                      "already_complete": sum(r["status"] == "existing" for r in results.values()),
                      "repair_failures": [r for r in results.values() if r["status"] == "failed"],
                      "pending": len(pending), "unsubmitted": len(set(failures) - submitted),
                      "halted": halted, "results": list(results.values())}
            write_json(args.summary, report)
            if not pending and (halted or (not producer_alive and not report["unsubmitted"])):
                if not halted and args.batch_log.stat().st_size != len(batch_text.encode()):
                    continue  # Drain a final failure written while the producer exited.
                print(json.dumps({"event": "finished", **{k: v for k, v in report.items() if k != "results"}}), flush=True)
                return int(bool(halted or report["repair_failures"]))
            if not pending and not report["unsubmitted"]:
                time.sleep(args.poll_seconds)


if __name__ == "__main__":
    sys.exit(main())
