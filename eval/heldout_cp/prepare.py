"""Compute and freeze a target's initial geometry and baselines before its CP."""

import gc
import json
from datetime import datetime, timezone

from eval.heldout_cp import protocol as p
from eval.heldout_cp import runtime


def prepare_dataset(doc, dataset_id, *, cache_dir, num_workers):
    import lightning as pl
    import torch
    from stable_cp.data.heldout import load_heldout_split, exact_train_indices

    dataset = p.DATASETS[dataset_id]
    gpu = runtime.check_environment()
    software = runtime.software()
    train = load_heldout_split(dataset, "train", str(cache_dir))
    indices_by_seed = {
        seed: exact_train_indices(train["label"], 1000, seed) for seed in p.SEEDS
    }
    data_by_seed = {
        seed: runtime.dataset_record(dataset, cache_dir, indices)
        for seed, indices in indices_by_seed.items()
    }
    for encoder in p.ENCODER_ORDER:
        lock = p.root_path(doc) / "pre" / encoder / dataset / "prepare.json"
        with p.seed_lock(lock):
            args = runtime.evaluation_args(encoder, dataset, 42, cache_dir, num_workers)
            pl.seed_everything(42, workers=True)
            model, device, config, weights_hash = runtime.load_model(encoder, args)
            path = p.geometry_path(doc, encoder, dataset)
            expected = dict(
                p.identity(doc, encoder, dataset),
                software=software,
                pretrained_weights_sha256=weights_hash,
                train_source_sha256=data_by_seed[42]["source_content_sha256"]["train"],
                train_pool_indices_sha256=data_by_seed[42]["train_pool_indices_sha256"],
            )
            if path.exists():
                p.check_identity(json.loads(path.read_text()), expected)
            else:
                geometry = runtime.initial_geometry(model, device, config, args)
                p.atomic_json(
                    path, dict(expected, status="complete", gpu=gpu, **geometry)
                )
                print(
                    f"GEOMETRY {encoder} {dataset} U={geometry['uniformity_t2']:.8f} "
                    f"n={geometry['n_geometry']}",
                    flush=True,
                )
            for seed in p.SEEDS:
                path = p.pre_path(doc, encoder, dataset, seed)
                indices = indices_by_seed[seed]
                data = data_by_seed[seed]
                expected = dict(
                    p.identity(doc, encoder, dataset, seed),
                    data=data,
                    software=software,
                    pretrained_weights_sha256=weights_hash,
                )
                if path.exists():
                    p.check_identity(
                        p.validate_pre(doc, encoder, dataset, seed), expected
                    )
                    print(
                        f"SKIP verified baseline {encoder} {dataset} seed={seed}",
                        flush=True,
                    )
                    continue
                args = runtime.evaluation_args(
                    encoder, dataset, seed, cache_dir, num_workers
                )
                print(f"BASELINE {encoder} {dataset} seed={seed} n=1000", flush=True)
                scores = runtime.evaluate(model, device, config, args, indices)
                row = dict(
                    expected,
                    status="complete",
                    gpu=gpu,
                    train_indices=indices,
                    completed_at_utc=datetime.now(timezone.utc).isoformat(),
                    **{f"pre_{key}": value for key, value in scores.items()},
                )
                p.check_metrics(row, "pre")
                p.atomic_json(path, row)
                p.validate_pre(doc, encoder, dataset, seed)
            del model
            gc.collect()
            torch.cuda.empty_cache()
    predictions = p.freeze_predictions(doc, dataset)
    print(f"FROZEN_INITIAL_GEOMETRY {predictions}", flush=True)
    print(
        f"PREPARED {dataset}: 2 encoders, 6 baseline evaluations, no CP/FT", flush=True
    )
