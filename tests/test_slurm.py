import os
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "run/slurm/run.sh"


@pytest.mark.parametrize("root_variable", ["CP_REPO_ROOT", "SLURM_SUBMIT_DIR"])
def test_slurm_forwards_arguments_and_exit_status(tmp_path, root_variable):
    root = tmp_path / "repo with spaces"
    root.mkdir()
    (root / "continued_pretraining.py").write_text(
        "import os, sys\nprint(os.getcwd())\nprint(repr(sys.argv[1:]))\nraise SystemExit(7)\n"
    )
    env = dict(os.environ, PYTHON=sys.executable)
    env.pop("CP_REPO_ROOT", None)
    env.pop("SLURM_SUBMIT_DIR", None)
    env[root_variable] = str(root)
    result = subprocess.run(
        ["bash", str(SCRIPT), "--run-name", "two words"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 7, result.stderr
    assert str(root) in result.stdout
    assert "['--run-name', 'two words']" in result.stdout


def test_slurm_shell_syntax():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
