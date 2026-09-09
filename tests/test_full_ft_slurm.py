import os
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARRAY = ROOT / "run/slurm/full_ft/array.sh"
SUBMIT = ROOT / "run/slurm/full_ft/submit.sh"


def test_array_dry_run_groups_three_seeds_and_uses_private_stage(tmp_path):
    manifest = tmp_path / "m.json"
    subprocess.run(["python3", str(ROOT / "eval/full_ft/manifest.py"), "--repo-root", str(ROOT),
                    "--methods", "DIET", "--budgets", "MAX", "--encoders", "SigLIP",
                    "--phases", "post", "--output", str(manifest)], check=True)
    env = dict(os.environ, FULL_FT_MANIFEST=str(manifest), SLURM_ARRAY_TASK_ID="0",
               FULL_FT_OUTDIR=str(tmp_path / "out"), FULL_FT_DATA_ROOT="/shared/data")
    run = subprocess.run(["bash", str(ARRAY), "--dry-run"], cwd=ROOT, env=env,
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "--seeds 42 43 44" in run.stdout
    assert "stable_datasets/processed/med_mnist/breastmnist-size=224" in run.stdout
    assert "full-ft-${SLURM_JOB_ID:-local}-${TASK_ID}" in ARRAY.read_text()


def test_submit_dry_run_has_first_wave_and_resource_overrides(tmp_path):
    env = dict(os.environ, FULL_FT_GRES="gpu:a100:2", FULL_FT_CONCURRENCY="7")
    run = subprocess.run(["bash", str(SUBMIT), "--dry-run", "--manifest-dir", str(tmp_path)],
                         cwd=ROOT, env=env, capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    out = run.stdout
    assert "--methods LeJEPA SimCLR DIET" in out
    assert "--budgets 500 MAX" in out
    assert "--encoders DINOv3 CLIP MAE SigLIP" in out
    assert "--phases post" in out
    path = next(Path(line.split()[-1]) for line in out.splitlines()
                if line.startswith("MANIFEST"))
    tasks = json.loads(path.read_text())["tasks"]
    assert len(tasks) == 315
    assert {task["phase"] for task in tasks} == {"post"}
    assert "--gres=gpu:a100:2" in out and "%7" in out
    assert "sbatch" in out


def test_array_restores_cluster_environment_and_private_staging():
    text = ARRAY.read_text()
    for directive in ["#SBATCH --partition=nvidia", "#SBATCH --account=civil",
                      "#SBATCH --nodes=1", "#SBATCH --ntasks-per-node=1"]:
        assert directive in text
    for fragment in ["module load miniconda/3-4.11.0", "conda activate env",
                     "PYTHONPATH", "PYTHONUNBUFFERED", "mktemp -d", "nvidia-smi"]:
        assert fragment in text
    assert "trap 'rm -rf --" in text
    assert 'LOCAL_CACHE=""' in text


def test_submit_independent_overrides_preserve_first_wave_scope(tmp_path):
    relative = Path(os.path.relpath(tmp_path, ROOT))
    post = subprocess.run(["bash", str(SUBMIT), "--dry-run", "--manifest-dir", str(relative),
                           "--phases", "post"], cwd=ROOT, capture_output=True, text=True)
    assert post.returncode == 0, post.stdout + post.stderr
    path = next(Path(line.split()[-1]) for line in post.stdout.splitlines()
                if line.startswith("MANIFEST"))
    assert path.is_absolute()
    assert len(json.loads(path.read_text())["tasks"]) == 315

    rooted = subprocess.run(["bash", str(SUBMIT), "--dry-run", "--manifest-dir", str(relative),
                             "--checkpoint-root", "/new/ckpts"], cwd=ROOT,
                            capture_output=True, text=True)
    assert rooted.returncode == 0, rooted.stdout + rooted.stderr
    path = next(Path(line.split()[-1]) for line in rooted.stdout.splitlines()
                if line.startswith("MANIFEST"))
    assert len(json.loads(path.read_text())["tasks"]) == 315


def test_full_grid_respects_post_filter_and_wall_time_never_exceeds_96h(tmp_path):
    run = subprocess.run(["bash", str(SUBMIT), "--dry-run", "--manifest-dir", str(tmp_path),
                          "--all"], cwd=ROOT, capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    path = next(Path(line.split()[-1]) for line in run.stdout.splitlines() if line.startswith("MANIFEST"))
    assert len(json.loads(path.read_text())["tasks"]) == 885
    invalid = subprocess.run(["bash", str(SUBMIT), "--dry-run", "--manifest-dir", str(tmp_path)],
                              cwd=ROOT, env=dict(os.environ, FULL_FT_TIME="96:01:00"),
                              capture_output=True, text=True)
    assert invalid.returncode != 0
    assert "SUBMIT" not in invalid.stdout


def test_spooled_array_uses_submit_directory_not_script_location(tmp_path):
    manifest = tmp_path / "m.json"
    subprocess.run(["python3", str(ROOT / "eval/full_ft/manifest.py"), "--repo-root", str(ROOT),
                    "--methods", "DIET", "--encoders", "SigLIP", "--phases", "post",
                    "--output", str(manifest)], check=True)
    spool = tmp_path / "slurm_script"
    for script in (ARRAY, ROOT / "run/slurm/cp-siglip/cp/diet_max_array.sh"):
        spool.write_text(script.read_text())
        env = dict(os.environ, SLURM_JOB_ID="12345", SLURM_SUBMIT_DIR=str(ROOT),
                   SLURM_ARRAY_TASK_ID="0", FULL_FT_MANIFEST=str(manifest),
                   SIGLIP_DIET_OUT_ROOT=str(tmp_path / "output"))
        result = subprocess.run(["bash", str(spool), "--dry-run"], cwd=tmp_path,
                                env=env, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "eval/full_ft/run.py" in result.stdout


def test_full_grid_submission_chains_chunks_with_mocked_sbatch(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "calls.jsonl"
    sbatch = bin_dir / "sbatch"
    sbatch.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "path = Path(os.environ['MOCK_SBATCH_CALLS'])\n"
        "previous = path.read_text().splitlines() if path.exists() else []\n"
        "with path.open('a') as handle:\n"
        "    handle.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "print(2100 + len(previous))\n"
    )
    sbatch.chmod(0o755)
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}",
               MOCK_SBATCH_CALLS=str(calls), FULL_FT_ARRAY_LIMIT="1000",
               FULL_FT_CONCURRENCY="12", FULL_FT_LOG_DIR=str(tmp_path / "logs"))
    result = subprocess.run(
        ["bash", str(SUBMIT), "--all", "--phases", "pre", "post",
         "--manifest-dir", str(tmp_path / "manifests")],
        cwd=ROOT, env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    commands = [json.loads(line) for line in calls.read_text().splitlines()]
    assert len(commands) == 2
    assert "--array=0-999%12" in commands[0]
    assert "--array=0-109%12" in commands[1]
    assert not any(arg.startswith("--dependency=") for arg in commands[0])
    assert "--dependency=afterany:2100" in commands[1]
    identities = []
    for command, count in zip(commands, (1000, 110)):
        exports = next(arg for arg in command if arg.startswith("--export="))
        fields = dict(field.split("=", 1) for field in exports.split(",")[1:])
        path = Path(fields["FULL_FT_MANIFEST"])
        assert path.is_absolute()
        tasks = json.loads(path.read_text())["tasks"]
        assert [task["task_id"] for task in tasks] == list(range(count))
        identities.extend(tuple(task[key] for key in
                                ("phase", "encoder", "method", "dataset", "budget"))
                          for task in tasks)
    assert len(set(identities)) == 1110
