"""The descriptor batch job must not depend on ambient conda activation."""

import os
from pathlib import Path
import subprocess

import pytest


SCRIPT = (Path(__file__).resolve().parents[1]
          / "run/slurm/cp-siglip/native-geometry/descriptor_baselines.sh")


def run_job(python, **overrides):
    env = dict(os.environ, DESCRIPTOR_PYTHON=str(python),
               SLURM_JOB_ID="test-job", SLURM_CPUS_PER_TASK="4", **overrides)
    env.pop("BASH_ENV", None)
    # Replace cluster-only commands while executing the actual job script.
    harness = """
module() { echo "Unexpected module activation" >&2; return 90; }
conda() { echo "Unexpected conda activation" >&2; return 91; }
python3() { echo "Unexpected ambient Python" >&2; return 92; }
cd() {
    if [ "$1" = /scratch/gs4133/zhd/CP/continued-pretraining ]; then
        return 0
    fi
    builtin cd "$@"
}
source "$1"
"""
    return subprocess.run(["bash", "--noprofile", "--norc", "-c",
                           harness, "descriptor-job-test", str(SCRIPT)],
                          env=env, capture_output=True, text=True, timeout=10)


@pytest.fixture
def python_stub(tmp_path):
    path = tmp_path / "environment with spaces" / "python3"
    path.parent.mkdir()
    path.write_text("""#!/bin/bash
printf 'SELECTED_PYTHON=%s\n' "$0"
printf 'ARG=%s\n' "$@"
if [ "${1:-}" = -c ]; then
    exit "${PREFLIGHT_EXIT:-0}"
fi
""")
    path.chmod(0o755)
    return path


def test_job_uses_pinned_python_without_conda_or_path_lookup(python_stub):
    result = run_job(python_stub)
    assert result.returncode == 0, result.stderr
    assert result.stdout.count(f"SELECTED_PYTHON={python_stub}") == 2
    assert "ARG=-m\nARG=eval.descriptor_baselines\nARG=siglip-native" in result.stdout
    assert "siglip_descriptor_baselines_v1/test-job" in result.stdout
    assert "ARG=--threads\nARG=4" in result.stdout
    assert "Unexpected" not in result.stderr


def test_job_stops_before_calculation_when_dependency_check_fails(python_stub):
    result = run_job(python_stub, PREFLIGHT_EXIT="17")
    assert result.returncode == 17, result.stderr
    assert "ARG=-m" not in result.stdout
    assert "RESULT_DIRECTORY=" not in result.stdout


@pytest.mark.parametrize("python", ["python3", "/missing/env/bin/python3"])
def test_job_rejects_unusable_interpreter(python):
    result = run_job(python)
    assert result.returncode == 2
    assert "executable absolute path" in result.stderr


def test_job_defaults_to_requested_gpu_partition():
    content = SCRIPT.read_text()
    assert "#SBATCH --partition=nvidia\n" in content
    assert "#SBATCH --qos=nvidia\n" in content
    assert "#SBATCH --gres=gpu:v100:1\n" in content
