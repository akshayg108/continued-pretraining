"""Rebuild frozen-evaluation cells from complete, identified seed records."""

import numpy as np
import pandas as pd


KEYS = ["encoder", "method", "dataset", "n"]
SEEDS = {42, 43, 44}
METHODS = {"LeJEPA": "LeJEPA-CP", "SimCLR": "SimCLR-CP", "DIET": "DIET-CP"}
RECOVERED_NAME = "main_grid_recovered_seeds_20260921.csv"


def complete_frozen_runs(reruns, recovered):
    fields = KEYS + ["seed", "post_knn", "post_lp"]
    for name, frame in (("reruns", reruns), ("recovered", recovered)):
        if not set(fields).issubset(frame.columns):
            raise ValueError(f"Missing seed-record fields in {name}.")
        if frame[KEYS + ["seed"]].isna().any().any():
            raise ValueError(f"Missing configuration or seed identifier in {name}.")
        scores = frame[["post_knn", "post_lp"]].to_numpy(dtype=float)
        if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
            raise ValueError(f"All {name} macro-F1 scores must be finite fractions.")
        if not frame.seed.isin(SEEDS).all():
            raise ValueError("Expected seed IDs 42, 43, 44.")
        if not frame.method.isin(METHODS).all():
            raise ValueError("Unknown CP method in seed records.")
        sizes = frame.n.to_numpy(dtype=float)
        if not np.isfinite(sizes).all() or ((sizes <= 0) | (sizes != np.floor(sizes))).any():
            raise ValueError("Invalid configuration size.")

    reruns = reruns.copy()
    recovered = recovered.copy()
    reruns["source"] = "rest_behavior.csv"
    reruns["seed_assignment"] = "explicit"
    if not {"source", "seed_assignment"}.issubset(recovered.columns):
        raise ValueError("Recovered records must identify their source and seed assignment.")
    if (recovered.source.isna().any() or recovered.source.eq("").any()
            or not recovered.seed_assignment.isin(["explicit", "unique_missing_seed"]).all()):
        raise ValueError("Invalid recovered-record provenance.")
    runs = pd.concat([frame for frame in (reruns, recovered) if len(frame)], ignore_index=True)
    if runs.duplicated(KEYS + ["seed"]).any():
        raise ValueError("Duplicate configuration/seed record.")
    old_keys = set(reruns[KEYS].itertuples(index=False, name=None))
    new_keys = set(recovered[KEYS].itertuples(index=False, name=None))
    if not new_keys.issubset(old_keys):
        raise ValueError("A recovered configuration is not part of the rerun audit.")
    for key, part in recovered.groupby(KEYS, sort=True):
        original = reruns
        for column, value in zip(KEYS, key):
            original = original[original[column].eq(value)]
        missing = SEEDS - set(original.seed)
        inferred = part[part.seed_assignment.eq("unique_missing_seed")]
        if len(inferred) and (len(missing) != 1 or set(inferred.seed) != missing):
            raise ValueError(f"Seed assignment requires exactly one missing seed: {key}.")
    for key, part in runs.groupby(KEYS, sort=True):
        if len(part) != 3 or set(part.seed) != SEEDS:
            raise ValueError(f"Expected all three seeds (42, 43, 44) for {key}.")
    runs["seed"] = runs.seed.astype(int)
    runs["n"] = runs.n.astype(int)
    return runs.sort_values(KEYS + ["seed"]).reset_index(drop=True)


def refresh_frozen_cells(archive, runs):
    """Update levels, sample SDs, and effects together; leave FT fields untouched."""
    output = archive.copy()
    cell_keys = ["Backbone", "Method", "dataset_key", "size"]
    if output.duplicated(cell_keys).any():
        raise ValueError("Duplicate configuration in the main-grid archive.")
    locations = {tuple(row): index for index, row in
                 zip(output.index, output[cell_keys].itertuples(index=False, name=None))}
    records = []
    for key, part in runs.groupby(KEYS, sort=True):
        encoder, method, dataset, size = key
        if len(part) != 3 or set(part.seed) != SEEDS:
            raise ValueError(f"Expected all three seeds (42, 43, 44) for {key}.")
        cell = (encoder, METHODS[method], dataset, size)
        if cell not in locations:
            raise ValueError(f"Unmatched main-grid configuration: {cell}.")
        index = locations[cell]
        record = dict(zip(KEYS, key))
        record.update(n_seeds=3, seeds="42;43;44",
                      n_archived_reruns=int(part.source.eq("rest_behavior.csv").sum()),
                      n_author_records=int(part.source.ne("rest_behavior.csv").sum()))
        for metric in ("knn", "lp"):
            baseline = float(output.at[index, f"{metric}_pre"])
            if not np.isfinite(baseline) or not 0 <= baseline <= 1:
                raise ValueError(f"Invalid archived baseline for {cell}.")
            if f"pre_{metric}" in part:
                references = part[f"pre_{metric}"].dropna().to_numpy(dtype=float)
                if not np.allclose(references, baseline, rtol=0, atol=1e-12):
                    raise ValueError(f"Mismatched archived baseline for {cell}.")
            values = part[f"post_{metric}"].to_numpy(dtype=float)
            if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
                raise ValueError(f"Invalid macro-F1 values for {cell}.")
            mean, sd = float(values.mean()), float(values.std(ddof=1))
            updates = {f"{metric}_post": mean, f"{metric}_post_s": sd,
                       f"d{metric}": mean - baseline}
            for column, value in updates.items():
                output.at[index, column] = value
            record.update({f"{metric}_pre": baseline, **updates})
        records.append(record)
    return output, pd.DataFrame(records)


def rebuild(archive, reruns, recovered):
    runs = complete_frozen_runs(reruns, recovered)
    output, summary = refresh_frozen_cells(archive, runs)
    # Export scores and provenance, not the historical rounded per-seed deltas.
    columns = KEYS + ["seed", "post_knn", "post_lp", "source", "seed_assignment"]
    if "results_json" in runs:
        columns.append("results_json")
    return output, runs[columns], summary
