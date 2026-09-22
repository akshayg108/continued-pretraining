"""Audited metric-roundoff recovery for an already frozen held-out CP manifest.

This entrypoint leaves the original implementation and strict artifact validator
unchanged. It clips only out-of-range evaluation scores within 1e-6 of [0, 1]
and records the raw scores and this wrapper's hash in each new metric artifact.
No training, feature extraction, geometry, or sampling settings are changed.
"""

import argparse
from contextlib import contextmanager
from pathlib import Path
import math
import sys

from eval.heldout_cp import protocol as p
from eval.heldout_cp import runtime


TOLERANCE = 1e-6
POLICY = "clip_out_of_range_metric_roundoff_v1"


def canonical_scores(raw):
    scores = {}
    for key in p.METRICS:
        value = raw.get(key)
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not -TOLERANCE <= value <= 1 + TOLERANCE
        ):
            raise ValueError(f"Invalid evaluation metric: {key}={value!r}")
        scores[key] = min(1.0, max(0.0, value)) if not 0 <= value <= 1 else value
    if set(raw) != set(p.METRICS):
        raise ValueError(f"Unexpected evaluation metrics: {sorted(set(raw) - set(p.METRICS))}")
    return scores


@contextmanager
def roundoff_evaluation(doc, phase):
    if phase not in ("pre", "post"):
        raise ValueError(f"Unknown evaluation phase: {phase}")
    evaluate, write = runtime.evaluate, p.atomic_json
    pending = {}
    provenance = dict(
        policy=POLICY,
        tolerance=TOLERANCE,
        wrapper_sha256=p.file_sha256(Path(__file__)),
        base_implementation_sha256=doc["implementation_sha256"],
        phase=phase,
    )

    def audited_evaluate(model, device, config, args, indices):
        raw = evaluate(model, device, config, args, indices)
        scores = canonical_scores(raw)
        key = (args.backbone, args.dataset, args.seed)
        pending[key] = (scores, dict(provenance, raw_scores=dict(raw)))
        for metric in p.METRICS:
            if raw[metric] != scores[metric]:
                print(
                    f"METRIC_ROUNDOFF dataset={args.dataset} seed={args.seed} "
                    f"phase={phase} metric={metric} raw={raw[metric]!r} "
                    f"saved={scores[metric]!r}",
                    flush=True,
                )
        return scores

    def audited_write(path, row):
        key = (row.get("model_id"), row.get("dataset"), row.get("seed"))
        if key in pending and any(f"{phase}_{metric}" in row for metric in p.METRICS):
            scores, audit = pending[key]
            if any(row.get(f"{phase}_{metric}") != scores[metric] for metric in p.METRICS):
                raise ValueError("Published metrics differ from the audited evaluation")
            row = dict(row, evaluation_numerics=audit)
        return write(path, row)

    runtime.evaluate, p.atomic_json = audited_evaluate, audited_write
    try:
        yield
    finally:
        runtime.evaluate, p.atomic_json = evaluate, write


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "fit"):
        item = sub.add_parser(command)
        item.add_argument("--manifest", type=Path, required=True)
        item.add_argument("--cache-dir", type=Path, default=Path("/scratch/gs4133/zhd/CP/data"))
        item.add_argument("--num-workers", type=int, default=8)
        item.add_argument("--dry-run", action="store_true")
        item.add_argument(
            "--dataset-id" if command == "prepare" else "--task-id", type=int, required=True
        )
        if command == "fit":
            item.add_argument("--seed", type=int, choices=p.SEEDS, required=True)
    args = parser.parse_args(argv)
    doc = p.load_manifest(args.manifest)
    if args.command == "fit" and not args.dry_run:
        if not 0 <= args.task_id < len(doc["tasks"]):
            parser.error("task-id must be 0..47")
        task = doc["tasks"][args.task_id]
        baseline = p.validate_pre(doc, task["encoder"], task["dataset"], args.seed)
        audit = baseline.get("evaluation_numerics")
        if audit is not None and (
            audit.get("wrapper_sha256") != p.file_sha256(Path(__file__))
            or audit.get("base_implementation_sha256") != doc["implementation_sha256"]
            or audit.get("policy") != POLICY
        ):
            raise ValueError("Metric recovery implementation changed since preparation")
    from eval.heldout_cp.__main__ import main as original_main

    print(
        f"METRIC_POLICY {POLICY} tolerance={TOLERANCE} "
        f"wrapper_sha256={p.file_sha256(Path(__file__))}",
        flush=True,
    )
    with roundoff_evaluation(doc, "pre" if args.command == "prepare" else "post"):
        original_main(argv)


if __name__ == "__main__":
    main()
