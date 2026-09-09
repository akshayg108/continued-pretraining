#!/usr/bin/env python3
"""Collect validated available-seed FT means and seed-paired pre/post deltas."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from eval.full_ft.run import IDENTITY_FIELDS, METRICS, SEEDS, load_tasks, validate_result


def _key(row):
    return tuple(row[k] for k in IDENTITY_FIELDS)


def _mean(values):
    return statistics.mean(values) if values else None


def _sd(values):
    return statistics.stdev(values) if len(values) > 1 else None


def _seeds(values):
    return ",".join(map(str, sorted(values)))


def summarize(outdir, tasks=None):
    groups = {_key(t): {} for t in tasks} if tasks is not None else {}
    for phase in ("pre", "post"):
        for path in sorted((Path(outdir) / phase).rglob("seed*.json")):
            row = json.loads(path.read_text())
            key = _key(row)
            if tasks is not None and key not in groups:
                continue
            seed = row["seed"]
            if seed not in SEEDS or row.get("status") not in {"success", "failed", "skipped_missing_checkpoint"}:
                raise ValueError(f"Invalid result status/seed: {path}")
            if seed in groups.setdefault(key, {}):
                raise ValueError(f"Duplicate result identity: {path}")
            if row["status"] == "success":
                validate_result(row)
            groups[key][seed] = row

    tables = {name: [] for name in ("per_seed", "summary", "availability", "paired_deltas")}
    for key, group in sorted(groups.items()):
        identity = dict(zip(IDENTITY_FIELDS, key))
        good = {seed: row for seed, row in group.items() if row["status"] == "success"}
        signatures = {(r["implementation_sha256"], json.dumps(r["software"], sort_keys=True)) for r in good.values()}
        if len(signatures) > 1:
            raise ValueError(f"Mixed FT code/environments within a mean: {identity}")
        missing = {s for s, r in group.items() if r["status"] == "skipped_missing_checkpoint"}
        failed = {s for s, r in group.items() if r["status"] == "failed"}
        availability = dict(identity, n_success=len(good), seeds_success=_seeds(good),
                            seeds_missing_checkpoint=_seeds(missing), seeds_failed=_seeds(failed),
                            seeds_unattempted=_seeds(set(SEEDS) - set(group)))
        tables["availability"].append(availability)
        mean_row = dict(availability)
        for metric in METRICS:
            values = [r[metric] for r in good.values()]
            mean_row[metric + "_mean"] = _mean(values)
            mean_row[metric + "_std"] = _sd(values)
        tables["summary"].append(mean_row)
        for seed, row in sorted(group.items()):
            tables["per_seed"].append({k: v for k, v in row.items() if not isinstance(v, (dict, list))})
        if identity["phase"] != "post":
            continue
        pre_identity = dict(identity, phase="pre", method="PRE")
        pre = groups.get(_key(pre_identity), {})
        paired = []
        for seed, post_row in good.items():
            pre_row = pre.get(seed)
            if pre_row is None or pre_row["status"] != "success":
                continue
            for field in ("train_indices_sha256", "test_labels_sha256", "n_train_actual", "n_test",
                          "implementation_sha256", "software", "sft_protocol"):
                if pre_row[field] != post_row[field]:
                    raise ValueError(f"Pre/post subset or protocol mismatch: {identity}, seed={seed}, {field}")
            paired.append((seed, pre_row, post_row))
        paired_row = dict(identity, n_pairs=len(paired), paired_seeds=_seeds(s for s, _, _ in paired))
        for metric in METRICS:
            values = [post[metric] - pre[metric] for _, pre, post in paired]
            paired_row["delta_" + metric.removeprefix("sft_") + "_mean"] = _mean(values)
            paired_row["delta_" + metric.removeprefix("sft_") + "_std"] = _sd(values)
        tables["paired_deltas"].append(paired_row)
    return tables


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Directory for four CSV tables")
    parser.add_argument("--manifest", type=Path, help="Include tasks that have not started yet")
    args = parser.parse_args(argv)
    tables = summarize(args.outdir, load_tasks(args.manifest) if args.manifest else None)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        fields = list(dict.fromkeys(k for row in rows for k in row)) or ["status"]
        path = args.output / (name + ".csv")
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"{path}: {len(rows)} rows")


if __name__ == "__main__":
    main()
