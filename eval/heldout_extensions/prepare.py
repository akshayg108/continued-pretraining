"""Prepare one encoder-target pair, retaining the original training subsets."""

from datetime import datetime, timezone
import gc
import json

from eval.heldout_extensions import protocol as p, runtime


def prepare(doc, task, *, cache_dir, num_workers):
    import lightning as pl
    import torch

    encoder, dataset = task["encoder"], task["dataset"]
    gpu = runtime.check_environment(task["gpu"])
    software = runtime.software()
    sources = {seed: p.source_baseline(doc, dataset, seed) for seed in p.SEEDS}
    for seed, source in sources.items():
        actual = runtime.dataset_record(dataset, cache_dir, source["train_indices"])
        if actual != source["data"]:
            raise ValueError(f"Original and current data differ: {dataset} seed={seed}")
    lock = p.root_path(doc) / "pre" / encoder / dataset / "prepare.json"
    with p.seed_lock(lock):
        args = runtime.evaluation_args(encoder, dataset, 42, cache_dir, num_workers)
        pl.seed_everything(42, workers=True)
        model, device, config, weights = runtime.load_model(encoder, args)
        expected = dict(p.identity(doc, encoder, dataset), software=software,
                        pretrained_weights_sha256=weights,
                        data_sha256=p.digest_json(sources[42]["data"]))
        path = p.geometry_path(doc, encoder, dataset)
        if path.exists():
            p.check_identity(json.loads(path.read_text()), expected)
        else:
            indices = p.source_geometry(doc, dataset)["geometry_indices"]
            geometry = runtime.initial_geometry(model, device, config, args, indices)
            p.atomic_json(path, dict(expected, status="complete", gpu=gpu, **geometry))
            print(f"GEOMETRY {encoder} {dataset} U={geometry['uniformity_t2']:.8f} n={geometry['n_geometry']}", flush=True)
        for seed, source in sources.items():
            path = p.pre_path(doc, encoder, dataset, seed)
            expected = dict(p.identity(doc, encoder, dataset, seed), data=source["data"],
                            software=software, pretrained_weights_sha256=weights)
            if path.exists():
                p.check_identity(p.validate_pre(doc, encoder, dataset, seed), expected)
                print(f"SKIP verified baseline {encoder} {dataset} seed={seed}", flush=True)
                continue
            args = runtime.evaluation_args(encoder, dataset, seed, cache_dir, num_workers)
            print(f"BASELINE {encoder} {dataset} seed={seed} n=1000", flush=True)
            scores, audit = runtime.evaluate(model, device, config, args, source["train_indices"])
            row = dict(expected, status="complete", gpu=gpu, train_indices=source["train_indices"],
                       evaluation_numerics=audit, completed_at_utc=datetime.now(timezone.utc).isoformat(),
                       **{f"pre_{key}": value for key, value in scores.items()})
            p.check_metrics(row, "pre")
            p.atomic_json(path, row)
            p.validate_pre(doc, encoder, dataset, seed)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        prediction = p.freeze_predictions(doc, encoder, dataset)
    print(f"PREPARED {encoder} {dataset}: 3 baselines, no CP/FT; predictions={prediction}", flush=True)
