"""Fresh public-weight CP fits, with encoder-specific dimensions and readouts."""

from datetime import datetime, timezone
import importlib
import json
import subprocess
import sys
import uuid

from eval.heldout_cp.run import training_args as base_training_args
from eval.heldout_extensions import protocol as p, runtime


def training_args(task, seed, cache_dir, num_workers, attempt):
    args = base_training_args(task, seed, cache_dir, num_workers, attempt)
    args.pool_strategy = task["pool"]
    return args


def fit_seed(doc, task, seed, *, cache_dir, num_workers):
    import lightning as pl
    from lightning.pytorch.loggers import WandbLogger
    from continued_pretraining import create_optim_config, run_training
    from stable_cp.data import create_transforms, create_train_datamodule

    gpu = runtime.check_environment(task["gpu"])
    prediction = p.freeze_predictions(doc, task["encoder"], task["dataset"])
    output = p.result_path(doc, task, seed)
    with p.seed_lock(output):
        if output.exists():
            p.validate_result(doc, task, seed, json.loads(output.read_text()))
            print(f"SKIP verified {task['encoder']} {task['method']} {task['dataset']} seed={seed}", flush=True)
            return
        baseline = p.validate_pre(doc, task["encoder"], task["dataset"], seed)
        software = runtime.software()
        if software != baseline["software"]:
            raise ValueError("Software changed between preparation and CP")
        indices = baseline["train_indices"]
        data_record = runtime.dataset_record(task["dataset"], cache_dir, indices)
        if data_record != baseline["data"]:
            raise ValueError("Training/evaluation samples differ from the prepared baseline")
        attempt = (p.root_path(doc) / "attempts" / task["encoder"] / task["method"]
                   / task["dataset"] / f"seed{seed}" / uuid.uuid4().hex)
        attempt.mkdir(parents=True, exist_ok=False)
        args = training_args(task, seed, cache_dir, num_workers, attempt)
        record = dict(p.identity(doc, task["encoder"], task["dataset"], seed),
                      method=task["method"], recipe=task["recipe"], software=software,
                      gpu=gpu, data=data_record, initialization="public_pretrained", no_ft=True,
                      predictions_sha256=p.file_sha256(prediction),
                      pre_sha256=p.file_sha256(p.pre_path(doc, task["encoder"], task["dataset"], seed)),
                      started_at_utc=datetime.now(timezone.utc).isoformat())
        p.atomic_json(attempt / "status.json", dict(record, status="running"))
        logger = None
        try:
            pl.seed_everything(seed, workers=True)
            model, device, config, weights = runtime.load_model(task["encoder"], args)
            if weights != baseline["pretrained_weights_sha256"]:
                raise ValueError("Public pretrained weights changed between preparation and CP")
            train_tf, val_tf = create_transforms(
                config, n_views={"LeJEPA": 8, "SimCLR": 2, "DIET": 1}[task["method"]],
                strong_aug=task["method"] != "DIET",
            )
            data, actual = create_train_datamodule(args, config, train_tf, val_tf, cache_dir, indices=indices)
            if actual != indices:
                raise ValueError("CP did not retain the prepared 1000-image subset")
            name = task["method"].lower()
            implementation = importlib.import_module(f"stable_cp.methods.{name}.{name}_cp")
            setup = getattr(implementation, f"setup_{name}")
            options = dict(task["recipe"], num_samples=1000, pool_strategy=task["pool"])
            optim = create_optim_config(args, args.warmup_epochs)
            dim = task["embed_dim"]
            if name == "lejepa":
                module = setup(model, dim, optim, implementation.build_sigreg_loss(args), **options)
            else:
                module = setup(model, dim, optim, **options)
            run_name = f"{task['encoder']}_{task['method']}_{task['dataset']}_n1000_s{seed}_{attempt.name[:8]}"
            logger = WandbLogger(project=p.PROTOCOL, name=run_name, log_model=False, save_dir=str(attempt))
            checkpoint = attempt / "checkpoints" / "cp.ckpt"
            print(f"CP_ONLY {run_name} gpu={gpu} pool={task['pool']} dim={dim} "
                  f"normalization={config['normalization']} recipe={task['recipe']} initialization=public_pretrained", flush=True)
            run_training(module, data, args, config, dim, args.freeze_epochs, logger, str(checkpoint), method=name)
            eval_args = runtime.evaluation_args(task["encoder"], task["dataset"], seed, cache_dir, num_workers)
            scores, audit = runtime.evaluate(model, device, config, eval_args, indices)
            row = dict(record, status="complete", pretrained_weights_sha256=weights,
                       checkpoint=str(checkpoint), checkpoint_sha256=p.file_sha256(checkpoint),
                       completed_at_utc=datetime.now(timezone.utc).isoformat(), evaluation_numerics=audit,
                       **{f"pre_{key}": baseline[f"pre_{key}"] for key in p.METRICS},
                       **{f"post_{key}": value for key, value in scores.items()})
            p.validate_result(doc, task, seed, row)
            for key, value in scores.items():
                logger.experiment.summary[f"final/{key}"] = value
                logger.experiment.summary[f"delta/{key}"] = value - baseline[f"pre_{key}"]
            logger.experiment.finish()
            logger = None
            p.atomic_json(output, row)
            p.atomic_json(attempt / "status.json", dict(row, result=str(output)))
            print(f"RESULT {output}", flush=True)
        except Exception as exc:
            p.atomic_json(attempt / "status.json", dict(record, status="failed", error=str(exc)))
            raise
        finally:
            if logger is not None:
                logger.experiment.finish()


def run_task(manifest, doc, task, *, cache_dir, num_workers):
    p.freeze_predictions(doc, task["encoder"], task["dataset"])
    for seed in task["seeds"]:
        subprocess.run([sys.executable, "-u", "-m", "eval.heldout_extensions", "fit",
                        "--manifest", str(manifest), "--task-id", str(task["task_id"]),
                        "--seed", str(seed), "--cache-dir", str(cache_dir),
                        "--num-workers", str(num_workers)], cwd=p.ROOT, check=True)
