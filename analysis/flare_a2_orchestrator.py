"""A2 orchestrator: runs analysis/flare_a2_worker.py for all 13 tasks
(parity already done separately) in parallel subprocesses, logging progress.

PYTHONPATH=src:analysis python analysis/flare_a2_orchestrator.py
"""
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

RESULTS = Path("analysis_outputs/final_results")
TASKS = [
    "unmarked-reversal", "marked-reversal", "odds-first", "bucket-sort",
    "binary-addition", "binary-multiplication", "compute-sqrt", "dyck-2-3",
    "repeat-01", "stack-manipulation", "missing-duplicate-string",
]
MAX_WORKERS = 3


def run_one(task):
    out_path = RESULTS / f"flare_a2_{task.replace('-', '_')}.json"
    log_path = RESULTS / f"flare_a2_{task.replace('-', '_')}.log"
    env = {**os.environ, "PYTHONPATH": "src:analysis", "OMP_NUM_THREADS": "2"}
    t0 = time.time()
    with open(log_path, "w") as logf:
        result = subprocess.run(
            [sys.executable, "analysis/flare_a2_worker.py", "--task", task],
            stdout=logf, stderr=subprocess.STDOUT, env=env,
        )
    elapsed = time.time() - t0
    ok = result.returncode == 0 and out_path.exists()
    return {"task": task, "ok": ok, "returncode": result.returncode, "elapsed_s": round(elapsed, 1)}


def main():
    print(f"=== A2 orchestrator: {len(TASKS)} tasks, {MAX_WORKERS} parallel workers ===", flush=True)
    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(run_one, t): t for t in TASKS}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            status = "OK" if r["ok"] else f"FAILED (rc={r['returncode']})"
            print(f"  [{len(results)}/{len(TASKS)}] {r['task']:28s} {status}  "
                  f"({r['elapsed_s']:.0f}s, {time.time()-t0:.0f}s total elapsed)", flush=True)
    n_ok = sum(1 for r in results if r["ok"])
    print(f"\n=== done: {n_ok}/{len(TASKS)} succeeded ===", flush=True)
    for r in results:
        if not r["ok"]:
            print(f"  FAILED: {r['task']} -- see analysis_outputs/final_results/flare_a2_{r['task'].replace('-','_')}.log", flush=True)


if __name__ == "__main__":
    main()
