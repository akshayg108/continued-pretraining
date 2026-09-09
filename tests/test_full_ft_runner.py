"""CPU regressions for checkpoint selection, seed resume, and paired summaries."""
import copy
import importlib
import json
from pathlib import Path

import pytest
import torch
from torch import nn


def module(name):
    return importlib.import_module(f"eval.full_ft.{name}")


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Linear(4, 4)
        self.blocks = nn.Sequential(nn.Linear(4, 4))
        self.attn_pool = nn.Linear(4, 4)


@pytest.mark.parametrize("prefix", ["", "backbone.", "backbone.vit.", "module.backbone.vit."])
def test_strict_checkpoint_preserves_every_backbone_tensor(tmp_path, prefix):
    checkpoint = module("checkpoint")
    source, target = Backbone(), Backbone()
    state = {prefix + k: v for k, v in source.state_dict().items()}
    state["diet_head.weight"] = torch.ones(500, 4)
    path = tmp_path / "cp.ckpt"
    torch.save({"state_dict": state}, path)
    audit = checkpoint.load_backbone_state(target, path)
    assert audit["prefix"] == prefix
    for key, value in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[key], value)


@pytest.mark.parametrize("damage", ["missing", "shape", "nan", "sft", "ambiguous"])
def test_corrupt_partial_or_sft_checkpoint_is_not_accepted(tmp_path, damage):
    checkpoint = module("checkpoint")
    model = Backbone()
    state = {"backbone." + k: v for k, v in model.state_dict().items()}
    key = "backbone.attn_pool.weight"
    if damage == "missing":
        del state[key]
    elif damage == "shape":
        state[key] = torch.ones(3, 3)
    elif damage == "nan":
        state[key] = torch.full_like(state[key], float("nan"))
    elif damage == "sft":
        state["classifier.weight"] = torch.ones(2, 4)
    else:
        state.update({"other." + k: v for k, v in model.state_dict().items()})
    path = tmp_path / "cp.ckpt"
    torch.save({"state_dict": state}, path)
    with pytest.raises(ValueError):
        checkpoint.load_backbone_state(model, path)


def test_mae_cp_can_omit_unused_native_clip_head_without_relaxing_pool_checks(tmp_path):
    checkpoint = module("checkpoint")
    source, target = Backbone(), Backbone()
    target.head = nn.Linear(4, 512)
    state = {"backbone.vit." + k: v for k, v in source.state_dict().items()}
    path = tmp_path / "cp.ckpt"
    torch.save({"state_dict": state}, path)
    checkpoint.discard_native_head(target)
    checkpoint.load_backbone_state(target, path)
    assert isinstance(target.head, nn.Identity)
    assert isinstance(target.attn_pool, nn.Linear)
    del state["backbone.vit.attn_pool.weight"]
    torch.save({"state_dict": state}, path)
    with pytest.raises(ValueError):
        checkpoint.load_backbone_state(target, path)


def task(tmp_path, phase="post"):
    model_id = "vit_base_patch16_clip_224.openai"
    paths = {}
    for seed in (42, 43, 44):
        path = tmp_path / f"dtd_{model_id}_n500_s{seed}.ckpt"
        if seed != 43:
            path.write_bytes(f"CP seed {seed}".encode())
        paths[str(seed)] = [str(path)] if phase == "post" else []
    return dict(task_id=0, phase=phase, scope="main", encoder="CLIP",
                method="SimCLR" if phase == "post" else "PRE", dataset="dtd",
                budget="500", n_samples=500, model_id=model_id, pool="cls",
                processed_subpath="dtd", checkpoints=paths)


def trainer(task, seed, checkpoint, **kwargs):
    return dict(sft_acc=0.6, sft_f1=seed / 100, sft_auroc=0.7,
                sft_protocol="full_ft_v1", sft_total_params=100,
                sft_trainable_params=100, n_train_actual=500, n_test=1880,
                train_indices_sha256="a" * 64, test_labels_sha256="b" * 64)


def test_missing_seed_is_recorded_and_successes_are_resumable(tmp_path, monkeypatch):
    run = module("run")
    selected = task(tmp_path)
    calls = []

    def train(*args, **kwargs):
        calls.append(args[1])
        return trainer(*args, **kwargs)

    monkeypatch.setattr(run, "train_one", train)
    output = tmp_path / "out"
    assert run.run_task(selected, outdir=output, cache_dir=tmp_path, device="cpu") == 0
    assert calls == [42, 44]
    skipped = json.loads(run.result_path(output, selected, 43).read_text())
    assert skipped["status"] == "skipped_missing_checkpoint"
    assert run.run_task(selected, outdir=output, cache_dir=tmp_path, device="cpu") == 0
    assert calls == [42, 44]
    assert not list(output.rglob("*.ckpt"))
    assert not list(output.rglob("*.pt"))


def test_bad_seed_does_not_discard_other_seeds_or_claim_success(tmp_path, monkeypatch):
    run = module("run")

    def train(selected, seed, checkpoint, **kwargs):
        if seed == 42:
            raise ValueError("invalid checkpoint tensors")
        return trainer(selected, seed, checkpoint, **kwargs)

    monkeypatch.setattr(run, "train_one", train)
    selected = task(tmp_path)
    assert run.run_task(selected, outdir=tmp_path / "out", cache_dir=tmp_path, device="cpu") == 1
    rows = [json.loads(run.result_path(tmp_path / "out", selected, s).read_text())
            for s in (42, 43, 44)]
    assert [r["status"] for r in rows] == ["failed", "skipped_missing_checkpoint", "success"]


@pytest.mark.parametrize("damage", ["metric_nan", "protocol", "frozen", "checkpoint_changed"])
def test_resume_is_fail_closed(tmp_path, monkeypatch, damage):
    run = module("run")
    selected = task(tmp_path)
    output = tmp_path / "out"
    monkeypatch.setattr(run, "train_one", trainer)
    assert run.run_task(selected, outdir=output, cache_dir=tmp_path, device="cpu", seeds=[42]) == 0
    path = run.result_path(output, selected, 42)
    row = json.loads(path.read_text())
    if damage == "metric_nan":
        row["sft_f1"] = float("nan")
    elif damage == "protocol":
        row["sft_protocol"] = "old_partial_ft"
    elif damage == "frozen":
        row["sft_trainable_params"] = 1
    else:
        Path(selected["checkpoints"]["42"][0]).write_bytes(b"changed checkpoint")
    path.write_text(json.dumps(row))
    assert run.run_task(selected, outdir=output, cache_dir=tmp_path, device="cpu", seeds=[42]) == 1


def test_manifest_rejects_wrong_model_or_wrong_seed_path(tmp_path):
    run = module("run")
    selected = task(tmp_path)
    for mutation in (dict(pool="mean"), dict(checkpoints={"42": selected["checkpoints"]["44"]})):
        broken = dict(selected, **mutation)
        with pytest.raises(ValueError):
            run.validate_task(broken)


def test_collector_pairs_only_matching_successful_seeds(tmp_path, monkeypatch):
    run, collect = module("run"), module("collect")
    output = tmp_path / "out"
    monkeypatch.setattr(run, "train_one", trainer)
    pre, post = task(tmp_path, "pre"), task(tmp_path)
    assert run.run_task(pre, outdir=output, cache_dir=tmp_path, device="cpu", seeds=[42]) == 0
    assert run.run_task(post, outdir=output, cache_dir=tmp_path, device="cpu") == 0
    summary = collect.summarize(output)
    post_mean = next(r for r in summary["summary"] if r["phase"] == "post")
    assert post_mean["n_success"] == 2
    assert post_mean["seeds_success"] == "42,44"
    assert post_mean["sft_f1_mean"] == pytest.approx(0.43)
    assert summary["paired_deltas"][0]["n_pairs"] == 1
    assert summary["paired_deltas"][0]["paired_seeds"] == "42"
    assert summary["paired_deltas"][0]["delta_f1_mean"] == pytest.approx(0.0)
    available = next(r for r in summary["availability"] if r["phase"] == "pre")
    assert available["seeds_unattempted"] == "43,44"


def test_collector_refuses_different_subsets_for_paired_delta(tmp_path, monkeypatch):
    run, collect = module("run"), module("collect")
    output = tmp_path / "out"

    def train(selected, *args, **kwargs):
        row = trainer(selected, *args, **kwargs)
        if selected["phase"] == "post":
            row["train_indices_sha256"] = "c" * 64
        return row

    monkeypatch.setattr(run, "train_one", train)
    for phase in ("pre", "post"):
        assert run.run_task(task(tmp_path, phase), outdir=output, cache_dir=tmp_path,
                            device="cpu", seeds=[42]) == 0
    with pytest.raises(ValueError, match="subset"):
        collect.summarize(output)
