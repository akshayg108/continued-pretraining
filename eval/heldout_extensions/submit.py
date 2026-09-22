"""Submit resource-grouped arrays with encoder-target-local preparation dependencies."""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

from eval.heldout_extensions import protocol as p


def submit(doc, manifest, *, cache_dir, log_dir, python, concurrency, dry_run):
    if not 1 <= concurrency <= 12:
        raise ValueError("concurrency must be 1..12")
    manifest, cache_dir, log_dir = (Path(v).resolve() for v in (manifest, cache_dir, log_dir))
    exports = dict(HELDOUT_PYTHON=python, HELDOUT_REPO_ROOT=str(p.ROOT),
                   HELDOUT_CACHE_DIR=str(cache_dir), HELDOUT_EXT_MANIFEST=str(manifest))
    if any("," in v or "\n" in v for v in exports.values()):
        raise ValueError("Slurm export paths must not contain commas or newlines")
    export = "ALL," + ",".join(f"{k}={v}" for k, v in exports.items())
    worker = p.ROOT / "run/slurm/heldout-extensions/worker.sh"
    prep_jobs, cp_jobs, held_arrays = {}, {}, []
    receipt = dict(manifest=str(manifest), preparations={}, cp={}, dependencies_configured=False, released=[])
    receipt_path = manifest.with_suffix(".submission.json")
    for stage, grid in (("prepare", doc["preparations"]), ("run", doc["tasks"])):
        for profile in ("v100", "a100"):
            id_key = "preparation_id" if stage == "prepare" else "task_id"
            ids = [row[id_key] for row in grid if row["gpu"] == profile]
            array = ",".join(map(str, ids))
            prefix = "heldout-ext-prep" if stage == "prepare" else "heldout-ext-cp"
            command = ["sbatch", "--parsable", f"--job-name={prefix}-{profile}",
                       f"--array={array}%{concurrency}", "--partition=nvidia", "--qos=nvidia", "--account=civil",
                       "--nodes=1", "--ntasks-per-node=1", f"--gres=gpu:{profile}:1", "--cpus-per-task=8",
                       "--mem=96G", "--time=96:00:00", f"--chdir={p.ROOT}",
                       f"--output={log_dir}/{prefix}-%A_%a.out", f"--error={log_dir}/{prefix}-%A_%a.err",
                       f"--export={export}"]
            if stage == "run":
                command.append("--hold")
            command.extend([str(worker), stage])
            print("SUBMIT " + shlex.join(command), flush=True)
            if dry_run:
                job = f"DRY_{stage}_{profile}"
            else:
                result = subprocess.run(command, check=True, text=True, capture_output=True)
                raw = result.stdout.strip()
                if not re.fullmatch(r"[0-9]+(?:;[^\s;]+)?", raw):
                    raise ValueError(f"Invalid sbatch job ID: {raw!r}; inspect the queue before retrying")
                job = raw.split(";", 1)[0]
                print(f"SUBMITTED stage={stage} gpu={profile} job={job}", flush=True)
            mapping = prep_jobs if stage == "prepare" else cp_jobs
            mapping.update({i: job for i in ids})
            receipt["preparations" if stage == "prepare" else "cp"][profile] = dict(job_id=job, indices=ids)
            if not dry_run:
                p.atomic_json(receipt_path, receipt)
            if stage == "run":
                held_arrays.append(job)
    try:
        for task in doc["tasks"]:
            i, prep = task["task_id"], task["preparation_id"]
            command = ["scontrol", "update", f"JobId={cp_jobs[i]}_{i}",
                       f"Dependency=afterok:{prep_jobs[prep]}_{prep}"]
            print("CONTROL " + shlex.join(command), flush=True)
            if not dry_run:
                subprocess.run(command, check=True)
    except subprocess.CalledProcessError:
        print(f"Dependency update failed. CP arrays remain held: {','.join(held_arrays)}. "
              "Do not release them until all dependencies are configured.", file=sys.stderr)
        raise
    receipt["dependencies_configured"] = True
    if not dry_run:
        p.atomic_json(receipt_path, receipt)
    for job in held_arrays:
        command = ["scontrol", "release", job]
        print("CONTROL " + shlex.join(command), flush=True)
        if not dry_run:
            subprocess.run(command, check=True)
            receipt["released"].append(job)
            p.atomic_json(receipt_path, receipt)
    print(f"{'DRY_RUN' if dry_run else 'SUBMITTED'} manifest={manifest}; "
          "16 preparation jobs, 48 CP jobs, 144 CP fits, 0 FT fits; QoS shares the user-wide 12-job limit", flush=True)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-base", type=Path,
                        default=Path(os.environ.get("HELDOUT_OUTPUT_BASE", str(p.DEFAULT_OUTPUT_BASE))))
    parser.add_argument("--cache-dir", type=Path,
                        default=Path(os.environ.get("HELDOUT_CACHE_DIR", "/scratch/gs4133/zhd/CP/data")))
    parser.add_argument("--concurrency", type=int, choices=range(1, 13), default=12)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.dry_run:
        for command in ("sbatch", "scontrol"):
            if shutil.which(command) is None:
                parser.error(f"Missing {command}; use --dry-run outside the cluster")
    doc = p.build_manifest(args.output_base, args.source_manifest)
    directory = args.output_base.resolve() / "heldout_extension_manifests"
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest = directory / f"heldout-extensions-{stamp}-{os.getpid()}.json"
    if manifest.exists():
        parser.error("Manifest already exists; refusing to overwrite")
    p.atomic_json(manifest, doc)
    manifest.chmod(0o444)
    log_dir = args.output_base.resolve() / "slurm-log/heldout-extensions"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"MANIFEST {manifest}", flush=True)
    submit(doc, manifest, cache_dir=args.cache_dir, log_dir=log_dir,
           python=sys.executable, concurrency=args.concurrency, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
