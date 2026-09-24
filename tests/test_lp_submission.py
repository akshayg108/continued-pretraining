"""Exercise LP-only submissions without contacting Slurm."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]


class LPSubmissionTests(unittest.TestCase):
    def submit(self, a100="132"):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binaries = root / "bin"
            binaries.mkdir()
            (root / "env/bin").mkdir(parents=True)
            python = root / "env/bin/python3"
            python.write_text(
                "#!/usr/bin/env bash\n"
                'if [[ "$*" == *"--gpu a100"* ]]; then\n'
                '    printf "%s\\n" "$TEST_A100"\n'
                "else\n    printf '1,69\\n'\nfi\n"
            )
            python.chmod(0o755)
            sbatch = binaries / "sbatch"
            sbatch.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, pathlib, sys\n"
                'path = pathlib.Path(os.environ["TEST_SUBMISSIONS"])\n'
                "lines = path.read_text().splitlines() if path.exists() else []\n"
                'with path.open("a") as f: f.write(json.dumps(sys.argv[1:]) + "\\n")\n'
                "print(700 + len(lines))\n"
            )
            sbatch.chmod(0o755)
            log = root / "submissions.jsonl"
            env = dict(
                os.environ,
                CP_ROOT=str(root),
                PATH=f"{binaries}:{os.environ['PATH']}",
                TEST_SUBMISSIONS=str(log),
                TEST_A100=a100,
            )
            result = subprocess.run(
                ["bash", str(REPO / "run/slurm/submit_lp_only.sh")],
                env=env,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            return [json.loads(line) for line in log.read_text().splitlines()]

    def test_flat_pre_grid_then_completed_post_groups_share_ten_slots(self):
        calls = self.submit()
        self.assertEqual(len(calls), 3)
        for call, array, gpu, phase in zip(
            calls,
            ("0-22%10", "1,69%10", "132%10"),
            ("v100", "v100", "a100"),
            ("pre", "post", "post"),
        ):
            self.assertIn(f"--array={array}", call)
            self.assertIn(f"--gres=gpu:{gpu}:1", call)
            self.assertEqual(call[call.index("--phase") + 1], phase)
            self.assertTrue(any(arg.endswith("/lp_only.sh") for arg in call))
        self.assertFalse(any(arg.startswith("--dependency") for arg in calls[0]))
        self.assertIn("--dependency=afterany:700", calls[1])
        self.assertIn("--dependency=afterany:701", calls[2])

    def test_no_a100_submission_without_completed_a100_checkpoints(self):
        self.assertEqual(len(self.submit(a100="")), 2)


if __name__ == "__main__":
    unittest.main()
