"""Collect paired extension results and complete-panel rank correlations."""

import json
from pathlib import Path
import statistics

from eval.heldout_cp.collect import write_csv
from eval.heldout_extensions import protocol as p


def collect(doc, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    records, summaries, groups, prediction_errors = [], [], {}, {}
    counts = dict(expected=sum(len(t["seeds"]) for t in doc["tasks"]), verified=0, missing=0, invalid=0)
    for prep in doc["preparations"]:
        key = prep["encoder"], prep["dataset"]
        try:
            if not p.predictions_path(doc, *key).is_file():
                raise ValueError("Initial geometry was not frozen before CP")
            p.freeze_predictions(doc, *key)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            prediction_errors[key] = str(exc)
    for task in doc["tasks"]:
        valid = []
        identity = {key: task[key] for key in ("encoder", "method", "dataset")}
        for seed in task["seeds"]:
            path = p.result_path(doc, task, seed)
            record = dict(identity, seed=seed, source=str(path), note="")
            if not path.is_file():
                record["status"] = "MISSING"
                counts["missing"] += 1
            else:
                try:
                    key = task["encoder"], task["dataset"]
                    if key in prediction_errors:
                        raise ValueError(prediction_errors[key])
                    row = p.validate_result(doc, task, seed, json.loads(path.read_text()))
                    record["status"] = "VERIFIED"
                    for metric in p.METRICS:
                        record[f"pre_{metric}"] = row[f"pre_{metric}"]
                        record[f"post_{metric}"] = row[f"post_{metric}"]
                        record[f"delta_{metric}"] = row[f"post_{metric}"] - row[f"pre_{metric}"]
                    valid.append(row)
                    counts["verified"] += 1
                except (ValueError, OSError, KeyError, TypeError) as exc:
                    record.update(status="INVALID", note=str(exc))
                    counts["invalid"] += 1
            records.append(record)
        summary = dict(identity, **p.summarize(valid), status="COMPLETE" if len(valid) == 3 else "INCOMPLETE")
        summaries.append(summary)
        groups[task["encoder"], task["method"], task["dataset"]] = summary
    correlations = []
    for encoder in p.ENCODER_ORDER:
        for method in (*p.METHODS, "MEAN_METHODS"):
            methods = p.METHODS if method == "MEAN_METHODS" else (method,)
            complete = [d for d in p.DATASETS if all(groups[encoder, m, d]["n_seeds"] == 3 for m in methods)]
            for metric in ("knn_f1", "linear_f1"):
                row = dict(encoder=encoder, method=method, metric=metric, n_datasets=len(complete),
                           spearman_rho=None, status="INCOMPLETE")
                if len(complete) == len(p.DATASETS):
                    from scipy.stats import spearmanr
                    geometry = [json.loads(p.geometry_path(doc, encoder, d).read_text())["uniformity_t2"] for d in complete]
                    effects = [statistics.mean(groups[encoder, m, d][f"delta_{metric}_mean"] for m in methods) for d in complete]
                    if len(set(geometry)) < 2 or len(set(effects)) < 2:
                        row["status"] = "CONSTANT"
                    else:
                        row.update(status="COMPLETE", spearman_rho=float(spearmanr(geometry, effects).statistic))
                correlations.append(row)
    fields = ["encoder", "method", "dataset", "seed", "status"]
    fields.extend(f"{phase}_{metric}" for metric in p.METRICS for phase in ("pre", "post", "delta"))
    write_csv(outdir / "seed_results.csv", records, [*fields, "source", "note"])
    write_csv(outdir / "summary.csv", summaries, list(summaries[0]))
    write_csv(outdir / "correlations.csv", correlations, list(correlations[0]))
    p.atomic_json(outdir / "status.json", counts)
    print(f"Verified {counts['verified']}/{counts['expected']} CP results; "
          f"missing={counts['missing']}, invalid={counts['invalid']}. Reports: {outdir.resolve()}")
    return counts
