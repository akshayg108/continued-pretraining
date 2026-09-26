"""Check the source launcher environment without Slurm, CUDA, or ML imports."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]


class SourcePretrainLauncherTests(unittest.TestCase):
    def check_launcher(self, stage, task, inherited, workers=None):
        with tempfile.TemporaryDirectory(prefix="source launcher ") as folder:
            root = Path(folder)
            repo = root / "repo"
            (repo / "run").mkdir(parents=True)
            shutil.copy2(REPO / "run/precp_env.sh", repo / "run/precp_env.sh")
            python = root / "env/bin/python3"
            python.parent.mkdir(parents=True)
            python.write_text(
                '#!/usr/bin/env bash\n'
                'printf "%s\\n" "${LD_LIBRARY_PATH-<unset>}"\n'
                'printf "%s\\n" "$@"\n'
            )
            python.chmod(0o755)
            env = {
                "PATH": os.environ["PATH"],
                "CP_ROOT": str(root),
                "CP_REPO_ROOT": str(repo),
                "SLURM_ARRAY_TASK_ID": str(task),
            }
            if inherited is not None:
                env["LD_LIBRARY_PATH"] = inherited
            if workers is not None:
                env["SOURCE_NUM_WORKERS"] = str(workers)
            result = subprocess.run(
                ["bash", str(REPO / "run/slurm/source_pretrain.sh"), stage],
                env=env, text=True, capture_output=True, check=True,
            )
            library_path, *args = result.stdout.splitlines()
            expected = [
                "-u", "run/source_pretrain.py", stage, "--root", str(root),
                "--imagenet-dir", str(root / "data/imagenet/train"),
                "--imagenet-val-dir", str(root / "data/imagenet_val"),
                "--output-dir", str(root / "outputs/source_coverage_v1"),
            ]
            if stage == "train":
                expected += [
                    "--condition", ("imagenet", "mixed")[task],
                    "--seed", "42", "--steps", "500400", "--resume",
                    "--num-workers", str(workers if workers is not None else 48),
                ]
            else:
                expected += ["--download-imagenet"]
            self.assertEqual(args, expected)
            suffix = f":{inherited}" if inherited else ""
            self.assertEqual(library_path, f"{root}/env/lib{suffix}")

    def test_training_uses_environment_runtime(self):
        for task in (0, 1):
            for inherited in (None, "", "/cluster/lib:/cuda/lib"):
                with self.subTest(task=task, inherited=inherited):
                    self.check_launcher("train", task, inherited)

    def test_preparation_uses_environment_runtime(self):
        for inherited in (None, "", "/cluster/lib:/cuda/lib"):
            with self.subTest(inherited=inherited):
                self.check_launcher("prepare", 0, inherited)

    def test_training_worker_override(self):
        for workers in (0, 24):
            with self.subTest(workers=workers):
                self.check_launcher("train", 0, None, workers=workers)

    def test_training_resource_defaults(self):
        script = (REPO / "run/slurm/source_pretrain.sh").read_text()
        directives = {
            line.removeprefix("#SBATCH ")
            for line in script.splitlines() if line.startswith("#SBATCH ")
        }
        self.assertTrue({
            "--gres=gpu:a100:1", "--constraint=80g", "--exclude=cn253,cn259",
            "--cpus-per-task=64", "--mem=128G", "--time=96:00:00",
        }.issubset(directives))
        self.assertNotIn("--gres=gpu:h200:1", directives)

    def test_submission_keeps_prepare_resources_separate(self):
        with tempfile.TemporaryDirectory(prefix="source submission ") as folder:
            root = Path(folder)
            repo = root / "repo"
            slurm = repo / "run/slurm"
            slurm.mkdir(parents=True)
            shutil.copy2(REPO / "run/precp_env.sh", repo / "run/precp_env.sh")
            for name in ("source_pretrain.sh", "submit_source_pretrain.sh"):
                shutil.copy2(REPO / "run/slurm" / name, slurm / name)
            binary = root / "bin"
            binary.mkdir()
            sbatch = binary / "sbatch"
            sbatch.write_text(
                f"#!{sys.executable}\n"
                "import json, os, sys\n"
                "with open(os.environ['SBATCH_CAPTURE'], 'a') as capture:\n"
                "    capture.write(json.dumps(sys.argv[1:]) + '\\n')\n"
                "print('12345;cluster')\n"
            )
            sbatch.chmod(0o755)
            capture = root / "sbatch.jsonl"
            subprocess.run(
                ["bash", str(slurm / "submit_source_pretrain.sh")],
                env={
                    "PATH": f"{binary}{os.pathsep}{os.environ['PATH']}",
                    "CP_ROOT": str(root),
                    "SBATCH_CAPTURE": str(capture),
                },
                text=True, capture_output=True, check=True,
            )
            prepare, train = [json.loads(line) for line in capture.read_text().splitlines()]
            for argument in (
                "--gres=gpu:v100:1", "--cpus-per-task=16", "--constraint=", "--exclude=",
            ):
                self.assertIn(argument, prepare)
            self.assertEqual(prepare[-2:], [str(slurm / "source_pretrain.sh"), "prepare"])
            self.assertIn("--dependency=afterok:12345", train)
            self.assertIn("--array=0-1%2", train)
            self.assertFalse(any(argument.startswith((
                "--gres=", "--cpus-per-task=", "--constraint=", "--exclude=",
            )) for argument in train))
            self.assertEqual(train[-2:], [str(slurm / "source_pretrain.sh"), "train"])


if __name__ == "__main__":
    unittest.main()
