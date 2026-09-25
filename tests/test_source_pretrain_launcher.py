"""Check the source launcher environment without Slurm, CUDA, or ML imports."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]


class SourcePretrainLauncherTests(unittest.TestCase):
    def check_launcher(self, stage, task, inherited):
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


if __name__ == "__main__":
    unittest.main()
