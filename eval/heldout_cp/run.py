"""Train every seed from public pretrained weights under the frozen protocol."""

from datetime import datetime, timezone
import importlib
import json
import subprocess
import sys
from types import SimpleNamespace
import uuid

from eval.heldout_cp import protocol as p
from eval.heldout_cp import runtime


def training_args(task, seed, cache_dir, num_workers, attempt):
    return SimpleNamespace(
        **task["recipe"],
        dataset=task["dataset"],
        n_samples=1000,
        seed=seed,
        cp_method=task["method"].lower(),
        backbone=task["model_id"],
        pool_strategy="cls",
        normalization_mode="pretrained",
        cache_dir=str(cache_dir),
        checkpoint_dir=str(attempt / "checkpoints"),
        num_workers=num_workers,
        resume=False,
        aggregation=False,
        skip_baseline=True,
        skip_final_eval=False,
        pre_cp_sft=False,
        post_cp_sft=False,
        no_cp=False,
        random_init=False,
        multivariate_test="slicing",
        univariate_test="epps_pulley",
        t_max=3.0,
        n_points=17,
        num_slices=1000,
        reduction="mean",
        clip_value=None,
    )


def fit_seed(doc, task, seed, *, cache_dir, num_workers):
    import lightning as pl
    from lightning.pytorch.loggers import WandbLogger
    from continued_pretraining import create_optim_config, run_training
    from stable_cp.data import create_transforms, create_train_datamodule

    gpu = runtime.check_environment()
    predictions = p.freeze_predictions(doc, task["dataset"])
    output = p.result_path(doc, task, seed)
    with p.seed_lock(output):
        if output.exists():
            p.validate_result(doc, task, seed, json.loads(output.read_text()))
            print(
                f"SKIP verified {task['encoder']} {task['method']} {task['dataset']} seed={seed}",
                flush=True,
            )
            return
        baseline = p.validate_pre(doc, task["encoder"], task["dataset"], seed)
        software = runtime.software()
        if software != baseline["software"]:
            raise ValueError("Software changed between preparation and CP")
        indices = baseline["train_indices"]
        data_record = runtime.dataset_record(task["dataset"], cache_dir, indices)
        if data_record != baseline["data"]:
            raise ValueError(
                "Training/evaluation samples differ from the prepared baseline"
            )
        attempt = (
            p.root_path(doc)
            / "attempts"
            / task["encoder"]
            / task["method"]
            / task["dataset"]
            / f"seed{seed}"
            / uuid.uuid4().hex
        )
        attempt.mkdir(parents=True, exist_ok=False)
        args = training_args(task, seed, cache_dir, num_workers, attempt)
        record = dict(
            p.identity(doc, task["encoder"], task["dataset"], seed),
            method=task["method"],
            recipe=task["recipe"],
            software=software,
            gpu=gpu,
            data=data_record,
            initialization="public_pretrained",
            no_ft=True,
            predictions_sha256=p.file_sha256(predictions),
            pre_sha256=p.file_sha256(
                p.pre_path(doc, task["encoder"], task["dataset"], seed)
            ),
            started_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        p.atomic_json(attempt / "status.json", dict(record, status="running"))
        logger = None
        try:
            pl.seed_everything(seed, workers=True)
            model, device, config, weights_hash = runtime.load_model(
                task["encoder"], args
            )
            if weights_hash != baseline["pretrained_weights_sha256"]:
                raise ValueError(
                    "Public pretrained weights changed between preparation and CP"
                )
            n_views = {"LeJEPA": 8, "SimCLR": 2, "DIET": 1}[task["method"]]
            train_tf, val_tf = create_transforms(
                config, n_views=n_views, strong_aug=task["method"] != "DIET"
            )
            data, actual = create_train_datamodule(
                args, config, train_tf, val_tf, cache_dir, indices=indices
            )
            if actual != indices:
                raise ValueError("CP did not retain the prepared 1000-image subset")
            module_name = task["method"].lower()
            implementation = importlib.import_module(
                f"stable_cp.methods.{module_name}.{module_name}_cp"
            )
            setup = getattr(implementation, f"setup_{module_name}")
            setup_args = dict(task["recipe"], num_samples=1000, pool_strategy="cls")
            optim = create_optim_config(args, args.warmup_epochs)
            if module_name == "lejepa":
                module = setup(
                    model,
                    768,
                    optim,
                    implementation.build_sigreg_loss(args),
                    **setup_args,
                )
            else:
                module = setup(model, 768, optim, **setup_args)
            name = f"{task['encoder']}_{task['method']}_{task['dataset']}_n1000_s{seed}_{attempt.name[:8]}"
            logger = WandbLogger(
                project=p.PROTOCOL, name=name, log_model=False, save_dir=str(attempt)
            )
            checkpoint = attempt / "checkpoints" / "cp.ckpt"
            print(
                f"CP_ONLY {name} normalization={config['normalization']} "
                "freeze=15 blocks=2 epochs=150 initialization=public_pretrained",
                flush=True,
            )
            run_training(
                module,
                data,
                args,
                config,
                768,
                args.freeze_epochs,
                logger,
                str(checkpoint),
                method=module_name,
            )
            eval_args = runtime.evaluation_args(
                task["encoder"], task["dataset"], seed, cache_dir, num_workers
            )
            scores = runtime.evaluate(model, device, config, eval_args, indices)
            row = dict(
                record,
                status="complete",
                pretrained_weights_sha256=weights_hash,
                checkpoint=str(checkpoint),
                checkpoint_sha256=p.file_sha256(checkpoint),
                completed_at_utc=datetime.now(timezone.utc).isoformat(),
                **{f"pre_{key}": baseline[f"pre_{key}"] for key in p.METRICS},
                **{f"post_{key}": value for key, value in scores.items()},
            )
            p.validate_result(doc, task, seed, row)
            for key, value in scores.items():
                logger.experiment.summary[f"final/{key}"] = value
                logger.experiment.summary[f"delta/{key}"] = (
                    value - baseline[f"pre_{key}"]
                )
            logger.experiment.finish()
            logger = None
            p.atomic_json(output, row)
            p.atomic_json(attempt / "status.json", dict(row, result=str(output)))
            print(f"RESULT {output}", flush=True)
        except Exception as exc:
            p.atomic_json(
                attempt / "status.json", dict(record, status="failed", error=str(exc))
            )
            raise
        finally:
            if logger is not None:
                logger.experiment.finish()


def run_task(manifest, doc, task, *, cache_dir, num_workers):
    p.freeze_predictions(doc, task["dataset"])
    for seed in task["seeds"]:
        command = [
            sys.executable,
            "-u",
            "-m",
            "eval.heldout_cp",
            "fit",
            "--manifest",
            str(manifest),
            "--task-id",
            str(task["task_id"]),
            "--seed",
            str(seed),
            "--cache-dir",
            str(cache_dir),
            "--num-workers",
            str(num_workers),
        ]
        subprocess.run(command, cwd=p.ROOT, check=True)
