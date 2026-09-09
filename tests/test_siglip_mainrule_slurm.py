import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARRAY = ROOT / "run/slurm/cp-siglip/mainrule/array.sh"
SUBMIT = ROOT / "run/slurm/cp-siglip/mainrule/submit.sh"


def run_submit(tmp_path, *args, env=None):
    merged = dict(os.environ, SIGLIP_MAINRULE_OUTPUT_BASE=str(tmp_path / "outputs"),
                  SIGLIP_MAINRULE_LOG_DIR=str(tmp_path / "logs"))
    if env:
        merged.update(env)
    return subprocess.run(["bash", str(SUBMIT), *args], cwd=ROOT, env=merged,
                          capture_output=True, text=True)


def test_submit_dry_run_builds_one_29_task_array_with_fixed_resources(tmp_path):
    result = run_submit(tmp_path, "--dry-run")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "python" in result.stdout and "eval.siglip_mainrule.protocol" in result.stdout
    assert "--array=0-28%12" in result.stdout
    for value in ("--gres=gpu:a100:1", "--cpus-per-task=8", "--mem=96G",
                  "--time=96:00:00"):
        assert value in result.stdout
    manifest = next(Path(token) for line in result.stdout.splitlines()
                    if line.startswith("MANIFEST") for token in line.split()
                    if token.endswith(".json"))
    assert len(json.loads(manifest.read_text())["tasks"]) == 29


def test_submit_calls_sbatch_exactly_once_and_exports_identity(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    calls = tmp_path / "calls.jsonl"
    sbatch = bindir / "sbatch"
    sbatch.write_text(
        f"#!{sys.executable}\nimport json,os,sys\n"
        "open(os.environ['CALLS'], 'a').write(json.dumps(sys.argv[1:])+'\\n')\n"
        "print('1234')\n"
    )
    sbatch.chmod(0o755)
    result = run_submit(tmp_path, "--concurrency", "7", env={
        "PATH": f"{bindir}:{os.environ['PATH']}", "CALLS": str(calls)})
    assert result.returncode == 0, result.stdout + result.stderr
    commands = [json.loads(line) for line in calls.read_text().splitlines()]
    assert len(commands) == 1
    command = commands[0]
    assert "--array=0-28%7" in command
    exports = next(arg for arg in command if arg.startswith("--export="))
    assert "SIGLIP_MAINRULE_MANIFEST=" in exports
    assert "SIGLIP_MAINRULE_REPO_ROOT=" in exports
    assert command[-1] == str(ARRAY)


def test_submit_rejects_recipe_flags_and_bad_concurrency(tmp_path):
    for args in (("--methods", "DIET"), ("--concurrency", "0"),
                 ("--concurrency", "13"), ("--concurrency", "abc"),
                 ("--output-base", "/tmp/wrong")):
        result = run_submit(tmp_path, *args)
        assert result.returncode == 2
        assert "SUBMIT" not in result.stdout


def test_array_has_fixed_resources_private_staging_and_no_old_artifact_paths():
    text = ARRAY.read_text()
    for fragment in ("#SBATCH --gres=gpu:a100:1", "#SBATCH --cpus-per-task=8",
                     "#SBATCH --mem=96G", "#SBATCH --time=96:00:00",
                     "module load miniconda/3-4.11.0", "conda activate env",
                     "set -euo pipefail", "mktemp -d", "df -Pk", "du -sk",
                     "trap 'rm -rf --", "nvidia-smi", "processed_subpath"):
        assert fragment in text
    assert "full_ft_v1" not in text
    assert "SIGLIP_DIET" not in text


def test_spooled_array_dry_run_uses_submit_dir_and_prints_three_seed_plan(tmp_path):
    manifest = tmp_path / "manifest.json"
    build = subprocess.run([
        sys.executable, "-m", "eval.siglip_mainrule.protocol",
        "--output-base", str(tmp_path / "out"), "--output", str(manifest)],
        cwd=ROOT, capture_output=True, text=True)
    assert build.returncode == 0, build.stdout + build.stderr
    spool = tmp_path / "slurm_script"
    spool.write_text(ARRAY.read_text())
    env = dict(os.environ, SLURM_JOB_ID="42", SLURM_SUBMIT_DIR=str(ROOT),
               SLURM_ARRAY_TASK_ID="0", SIGLIP_MAINRULE_MANIFEST=str(manifest),
               SIGLIP_MAINRULE_CACHE_DIR="/shared/cache")
    result = subprocess.run(["bash", str(spool), "--dry-run"], cwd=tmp_path,
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "eval.siglip_mainrule.run" in result.stdout
    for seed in ("42", "43", "44"):
        assert f'"seed": {seed}' in result.stdout
    assert not (tmp_path / "stage").exists()


def test_array_rejects_unknown_arguments_before_work():
    result = subprocess.run(["bash", str(ARRAY), "--not-a-real-flag"], cwd=ROOT,
                            capture_output=True, text=True)
    assert result.returncode == 2


def test_array_real_path_stages_for_runner_then_cleans_private_cache(tmp_path):
    cache = tmp_path / "shared"
    source = cache / "stable_datasets/processed/example/data"
    source.mkdir(parents=True)
    (source / "sample.bin").write_bytes(b"sample")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    marker = tmp_path / "runner.txt"
    fake_python = bindir / "python"
    fake_python.write_text(
        "#!/bin/bash\nset -euo pipefail\n"
        "if [ \"${1:-}\" = - ]; then echo example/data; exit 0; fi\n"
        "if [ \"${1:-}\" = -c ]; then exit 0; fi\n"
        "cache=''\nwhile [ $# -gt 0 ]; do\n"
        "  if [ \"$1\" = --cache-dir ]; then cache=$2; shift 2; else shift; fi\n"
        "done\n"
        "test -f \"${cache}/stable_datasets/processed/example/data/sample.bin\"\n"
        "printf '%s\\n' \"${cache}\" > \"${RUNNER_MARKER}\"\n"
    )
    fake_python.chmod(0o755)
    nvidia_smi = bindir / "nvidia-smi"
    nvidia_smi.write_text("#!/bin/bash\nexit 0\n")
    nvidia_smi.chmod(0o755)
    stage_root = tmp_path / "stage"
    stage_root.mkdir()
    fake_repo = tmp_path / "repo"
    (fake_repo / "eval/siglip_mainrule").mkdir(parents=True)
    env = dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}",
               PYTHON=str(fake_python), RUNNER_MARKER=str(marker),
               TMPDIR=str(stage_root), SIGLIP_MAINRULE_SKIP_ENV_SETUP="1",
               SIGLIP_MAINRULE_REPO_ROOT=str(fake_repo),
               SIGLIP_MAINRULE_MANIFEST=str(manifest), SLURM_ARRAY_TASK_ID="0",
               SIGLIP_MAINRULE_CACHE_DIR=str(cache))
    result = subprocess.run(["bash", str(ARRAY)], cwd=ROOT, env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    private_cache = Path(marker.read_text().strip())
    assert private_cache.parent == stage_root
    assert not private_cache.exists()
