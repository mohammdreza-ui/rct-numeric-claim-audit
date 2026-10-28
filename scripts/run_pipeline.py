#!/usr/bin/env python3
"""Automated pipeline runner with dedicated run directory, isolated logs, and run archiving.

Usage:
    python scripts/run_pipeline.py --pool 3000 --name run_full_300
    python scripts/run_pipeline.py --pool 1500 --name run_pilot_150
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Ensure ARGV0 does not interfere with venv in AppImage
os.environ.pop("ARGV0", None)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

def run_step(cmd: list[str], step_num: int, step_name: str, log_dir: Path, master_log: Path) -> None:
    step_log = log_dir / f"{step_num:02d}_{step_name}.log"
    print(f"\n{'='*70}")
    print(f"▶ [{step_num}/8] Starting: {step_name.upper()}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Log:     {step_log.relative_to(ROOT)}")
    print(f"{'='*70}\n", flush=True)

    t0 = time.time()
    with open(step_log, "w", encoding="utf-8") as f_step, open(master_log, "a", encoding="utf-8") as f_master:
        header = f"\n=== STEP {step_num}: {step_name} [{datetime.datetime.now().isoformat()}] ===\n"
        f_master.write(header)
        
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=os.environ,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            f_step.write(line)
            f_master.write(line)
        
        ret = proc.wait()
        dt = time.time() - t0
        footer = f"\n=== COMPLETED in {dt:.1f}s with returncode {ret} ===\n"
        f_step.write(footer)
        f_master.write(footer)
        
        if ret != 0:
            print(f"\n❌ ERROR: Step {step_name} failed with code {ret} after {dt:.1f}s.")
            print(f"Check log: {step_log}")
            sys.exit(ret)
        else:
            print(f"\n✅ Step {step_name} completed in {dt:.1f}s.")


def main():
    parser = argparse.ArgumentParser(description="Run pipeline with structured logging and dedicated run folder.")
    parser.add_argument("--pool", type=int, default=3000, help="PubMed search pool size (e.g. 1500 for ~150 abstracts, 3000 for ~300)")
    parser.add_argument("--name", type=str, default="", help="Custom name for this run (e.g. run_full_300)")
    parser.add_argument("--generator", type=str, default="local", help="Generator name in configs/config.toml")
    parser.add_argument("--samples", type=int, default=3, help="Self-consistency samples per abstract")
    parser.add_argument("--labels", type=int, default=100, help="Claims to sample for label sheet")
    args = parser.parse_args()

    # Determine run name and directory
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.name or f"run_{args.pool}pool_{now_str}"
    run_dir = ROOT / "runs" / run_name
    logs_dir = run_dir / "logs"
    work_backup = run_dir / "work"
    results_backup = run_dir / "results"
    
    logs_dir.mkdir(parents=True, exist_ok=True)
    work_backup.mkdir(parents=True, exist_ok=True)
    results_backup.mkdir(parents=True, exist_ok=True)

    master_log = logs_dir / "pipeline_all.log"

    # Backup current config used
    cfg_file = ROOT / "configs" / "config.toml"
    shutil.copy2(cfg_file, run_dir / "config_snapshot.toml")

    # Record metadata
    meta = {
        "run_name": run_name,
        "started_at": datetime.datetime.now().isoformat(),
        "pool": args.pool,
        "generator": args.generator,
        "samples": args.samples,
        "labels": args.labels,
    }
    (run_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    py = sys.executable

    print(f"\n🚀 STARTING PIPELINE RUN: {run_name}")
    print(f"📂 Output directory: {run_dir.relative_to(ROOT)}")
    print(f"📝 Full log file:   {master_log.relative_to(ROOT)}")

    # 1. Fetch
    run_step([py, "-m", "rctaudit", "fetch", "--pool", str(args.pool)], 1, "fetch", logs_dir, master_log)

    # 2. Generate
    gen_cmd = [py, "-m", "rctaudit", "generate", "--generator", args.generator, "--samples", str(args.samples), "--logprobs"]
    run_step(gen_cmd, 2, "generate", logs_dir, master_log)

    # 3. Extract
    run_step([py, "-m", "rctaudit", "extract", "--generator", args.generator], 3, "extract", logs_dir, master_log)

    # 4. Verify
    run_step([py, "-m", "rctaudit", "verify", "--generator", args.generator], 4, "verify", logs_dir, master_log)

    # 5. Signals
    run_step([py, "-m", "rctaudit", "signals", "--generator", args.generator], 5, "signals", logs_dir, master_log)

    # 6. Verbalized
    run_step([py, "-m", "rctaudit", "verbalized", "--generator", args.generator], 6, "verbalized", logs_dir, master_log)

    # 7. P(Yes) verifier-style confidence (needs top_logprobs; empty column if the server lacks it)
    run_step([py, "-m", "rctaudit", "pyes", "--generator", args.generator], 7, "pyes", logs_dir, master_log)

    # 8. Labelsheet
    run_step([py, "-m", "rctaudit", "labelsheet", "--generator", args.generator, "--n", str(args.labels)], 8, "labelsheet", logs_dir, master_log)

    # Copy artifacts to run directory
    print("\n📦 Archiving all run artifacts into run directory...")
    for f in (ROOT / "work").glob("*.jsonl"):
        shutil.copy2(f, work_backup / f.name)
    for f in (ROOT / "work").glob("*.csv"):
        shutil.copy2(f, work_backup / f.name)
    for f in (ROOT / "results").glob("*.csv"):
        shutil.copy2(f, results_backup / f.name)
    if (ROOT / "data" / "manifest.csv").exists():
        shutil.copy2(ROOT / "data" / "manifest.csv", run_dir / "manifest.csv")

    meta["completed_at"] = datetime.datetime.now().isoformat()
    (run_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    print(f"\n🎉 ALL STEPS COMPLETED SUCCESSFULLY!")
    print(f"📁 All files, datasets, and step logs are preserved in:")
    print(f"   {run_dir}")


if __name__ == "__main__":
    main()
