import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "eval" / "full_ft" / "manifest.py"


def load_manifest_module():
    spec = importlib.util.spec_from_file_location("full_ft_manifest", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_full_source_grid_coverage_and_shared_pre_deduplication():
    tasks = load_manifest_module().build_tasks(ROOT, phases=("pre", "post"))
    main_post = [t for t in tasks if t["scope"] == "main" and t["phase"] == "post"]
    main_pre = [t for t in tasks if t["scope"] == "main" and t["phase"] == "pre"]
    siglip_post = [t for t in tasks if t["scope"] == "siglip" and t["phase"] == "post"]
    siglip_pre = [t for t in tasks if t["scope"] == "siglip" and t["phase"] == "pre"]

    assert len(main_post) == 840
    assert len(main_pre) == 210
    assert len(siglip_post) == 45
    assert len(siglip_pre) == 15
    assert {t["method"] for t in main_post} == {"DIET", "LeJEPA", "MAE", "SimCLR"}
    assert {t["method"] for t in siglip_post} == {"DIET", "LeJEPA", "SimCLR"}
    assert {t["method"] for t in main_pre + siglip_pre} == {"PRE"}
    assert len({(t["scope"], t["encoder"], t["dataset"], t["budget"])
                for t in main_pre + siglip_pre}) == len(main_pre + siglip_pre)
    assert [t["task_id"] for t in tasks] == list(range(len(tasks)))


def test_default_manifest_never_schedules_pre_ft():
    module = load_manifest_module()
    tasks = module.build_tasks(ROOT)
    assert len(tasks) == 885
    assert {task["phase"] for task in tasks} == {"post"}
    first_wave = module.build_tasks(ROOT, methods=["LeJEPA", "SimCLR", "DIET"],
                                    budgets=["500", "MAX"])
    assert len(first_wave) == 315
    assert {task["phase"] for task in first_wave} == {"post"}


def test_budget_aliases_and_dataset_specific_maxima():
    tasks = load_manifest_module().build_tasks(ROOT, phases=("post",))
    main = [t for t in tasks if t["scope"] == "main"]
    aliases = {"food101": 101, "flowers102": 102, "cars196": 196, "cub200": 200}
    for dataset, actual in aliases.items():
        assert any(t["dataset"] == dataset and t["budget"] == "100"
                   and t["n_samples"] == actual for t in main)
    assert not any(t["dataset"] == "flowers102" and t["budget"] == "1000" for t in main)
    aircraft = [t for t in tasks if t["dataset"] == "fgvc_aircraft" and t["budget"] == "MAX"]
    assert aircraft and {t["n_samples"] for t in aircraft} == {3334}


def test_filters_are_applied_before_contiguous_ids_and_keep_pre_shared():
    tasks = load_manifest_module().build_tasks(
        ROOT, methods=["LeJEPA", "SimCLR"], budgets=["MAX"],
        encoders=["SigLIP"], phases=["pre", "post"])
    assert len(tasks) == 45
    assert sum(t["phase"] == "pre" for t in tasks) == 15
    assert sum(t["phase"] == "post" for t in tasks) == 30
    assert [t["task_id"] for t in tasks] == list(range(45))


def test_checkpoints_group_exact_seeds_inventory_first_and_relocate():
    new_root = Path("/mnt/relocated/ckpts")
    tasks = load_manifest_module().build_tasks(
        ROOT, methods=["DIET"], budgets=["MAX"], encoders=["CLIP"],
        phases=["post"], checkpoint_root=new_root)
    task = next(t for t in tasks if t["dataset"] == "breastmnist")
    assert list(task["checkpoints"]) == ["42", "43", "44"]
    assert all(paths and str(paths[0]).startswith(str(new_root))
               for paths in task["checkpoints"].values())
    assert task["checkpoints"]["42"][0].endswith(
        "/cp/DIET/pretrained/BreastMNIST/CLIP/cp/"
        "breastmnist_vit_base_patch16_clip_224.openai_n546_s42.ckpt")
    assert all("sft" not in p.lower() and "teacher" not in p.lower()
               for paths in task["checkpoints"].values() for p in paths)


def test_siglip_diet_uses_launcher_layout_and_pre_has_seed_slots():
    tasks = load_manifest_module().build_tasks(
        ROOT, methods=["DIET"], budgets=["MAX"], encoders=["SigLIP"], phases=["pre", "post"])
    post = next(t for t in tasks if t["phase"] == "post" and t["dataset"] == "food101")
    assert post["checkpoints"]["44"][0].endswith(
        "/cp-siglip/cp/DIET/Food101/SigLIP/cp/"
        "food101_vit_base_patch16_siglip_224.v2_webli_n75750_s44.ckpt")
    pre = next(t for t in tasks if t["phase"] == "pre" and t["dataset"] == "food101")
    assert pre["checkpoints"] == {"42": [], "43": [], "44": []}


def test_launch_fallback_includes_exact_budget_directory_for_missing_seed():
    tasks = load_manifest_module().build_tasks(
        ROOT, methods=["LeJEPA"], budgets=["MAX"], encoders=["CLIP"], phases=["post"])
    food = next(t for t in tasks if t["dataset"] == "food101")
    expected = ("/cp/LeJEPA/pretrained/Food101/CLIP/all/cp/"
                "food101_vit_base_patch16_clip_224.openai_n75750_s44.ckpt")
    assert any(path.endswith(expected) for path in food["checkpoints"]["44"])


def test_cli_writes_versioned_wrapper_and_accepts_nargs_filters(tmp_path):
    output = tmp_path / "manifest.json"
    subprocess.run(
        [sys.executable, str(MODULE_PATH), "--repo-root", str(ROOT),
         "--methods", "LeJEPA", "SimCLR", "--budgets", "500", "MAX",
         "--encoders", "DINOv3", "CLIP",
         "--output", str(output)], check=True)
    doc = json.loads(output.read_text())
    assert doc["schema_version"] == 1
    assert isinstance(doc["source"], dict)
    assert doc["tasks"]
    assert {t["method"] for t in doc["tasks"]} == {"LeJEPA", "SimCLR"}
    assert {t["budget"] for t in doc["tasks"]} <= {"500", "MAX"}
    assert {t["encoder"] for t in doc["tasks"]} <= {"DINOv3", "CLIP"}
    assert {t["phase"] for t in doc["tasks"]} == {"post"}
