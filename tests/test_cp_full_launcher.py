"""Exercise CP array submission without Slurm, GPUs, or ML dependencies."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]
MOCK_COMMAND = """import json
import os
from pathlib import Path
import sys

name = Path(sys.argv[0]).name
args = sys.argv[1:]
record = {"name": name, "args": args, "environment": {
    key: os.environ.get(key) for key in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "LD_LIBRARY_PATH",
    )
}}
path = Path(os.environ["CALLS_FILE"])
with path.open("a") as stream:
    stream.write(json.dumps(record) + "\\n")

if name == "squeue":
    print(os.environ.get("EXISTING_JOBS", ""))
elif name == "sbatch":
    calls = [json.loads(line) for line in path.read_text().splitlines()]
    print(f"{7000 + sum(call['name'] == 'sbatch' for call in calls)};cluster")
elif "-c" in args:
    print("CP runtime imports OK")
elif "array" in args:
    group = args[args.index("--group") + 1] if "--group" in args else "small"
    gpu = args[args.index("--gpu") + 1]
    print(json.loads(os.environ["TASK_LISTS"])[group].get(gpu, ""))
"""


class CPFullLauncherTests(unittest.TestCase):
    def launch(self, *, env_overrides=None, tasks=None, args=()):
        with tempfile.TemporaryDirectory(prefix="cp launcher ") as folder:
            root = Path(folder)
            repo = root / "repo"
            slurm = repo / "run/slurm"
            slurm.mkdir(parents=True)
            shutil.copy2(REPO / "run/precp_env.sh", repo / "run/precp_env.sh")
            shutil.copy2(REPO / "run/slurm/submit_cp_full.sh", slurm)
            calls_file = root / "calls.jsonl"
            bin_dir = root / "bin"
            bin_dir.mkdir()
            python = root / "env/bin/python3"
            python.parent.mkdir(parents=True)
            for command in (python, bin_dir / "squeue", bin_dir / "sbatch"):
                command.write_text(f"#!{sys.executable}\n{MOCK_COMMAND}")
                command.chmod(0o755)
            env = {
                "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
                "CP_ROOT": str(root),
                "CALLS_FILE": str(calls_file),
                "EXISTING_JOBS": "6002\n6001\n6002",
                "LD_LIBRARY_PATH": "/cluster/lib:/cuda/lib",
                "TASK_LISTS": json.dumps(
                    tasks
                    or {
                        "small": {"v100": "1,2,3,177", "a100": "0,4,176"},
                        "four-block": {"a100": "220,221,222,297", "a100-80gb": "268,272,283"},
                    }
                ),
            }
            env.update(env_overrides or {})
            result = subprocess.run(
                ["bash", str(slurm / "submit_cp_full.sh"), *args],
                env=env,
                text=True,
                capture_output=True,
            )
            calls = (
                [json.loads(line) for line in calls_file.read_text().splitlines()]
                if calls_file.exists()
                else []
            )
            return result, calls, root, repo

    def submissions(self, calls):
        return [call["args"] for call in calls if call["name"] == "sbatch"]

    def assert_success(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_default_small_group_keeps_v100_then_a100(self):
        result, calls, root, repo = self.launch()
        self.assert_success(result)
        submissions = self.submissions(calls)
        self.assertEqual(len(submissions), 2)
        for submission, gpu, tasks in zip(
            submissions, ("v100", "a100"), ("1,2,3,177", "0,4,176")
        ):
            self.assertIn(f"--gres=gpu:{gpu}:1", submission)
            self.assertIn(f"--array={tasks}%10", submission)
            self.assertIn(f"--output={root}/outputs/slurm-log/cp-full-{gpu}-%A_%a.out", submission)
            self.assertIn("--job-name=cp-full", submission)
            self.assertEqual(submission[-1], str(repo / "run/slurm/cp_full.sh"))
            self.assertFalse(
                any(arg.startswith(("--constraint=", "--exclude=")) for arg in submission)
            )
        self.assertIn("--dependency=afterany:6001:6002", submissions[0])
        self.assertIn("--dependency=afterany:7001", submissions[1])
        for call in calls:
            if call["name"] == "python3" and "-c" not in call["args"]:
                self.assertIn("--group", call["args"])
                self.assertEqual(call["args"][call["args"].index("--group") + 1], "small")

    def test_four_block_resources_logs_task_lists_and_order(self):
        result, calls, root, _ = self.launch(env_overrides={"CP_GROUP": "four-block"})
        self.assert_success(result)
        submissions = self.submissions(calls)
        self.assertEqual(len(submissions), 2)
        for submission, gpu, tasks in zip(
            submissions, ("a100", "a100-80gb"), ("220,221,222,297", "268,272,283")
        ):
            self.assertIn("--gres=gpu:a100:1", submission)
            self.assertIn(f"--array={tasks}%10", submission)
            self.assertIn(
                f"--output={root}/outputs/slurm-log/cp-four-block-{gpu}-%A_%a.out", submission
            )
            self.assertIn(
                f"--error={root}/outputs/slurm-log/cp-four-block-{gpu}-%A_%a.err", submission
            )
            self.assertIn("--job-name=cp-full", submission)
        self.assertFalse(
            any(arg.startswith(("--constraint=", "--exclude=")) for arg in submissions[0])
        )
        self.assertIn("--constraint=80g", submissions[1])
        self.assertIn("--exclude=cn253,cn259", submissions[1])
        self.assertIn("--dependency=afterany:6001:6002", submissions[0])
        self.assertIn("--dependency=afterany:7001", submissions[1])
        runner_calls = [
            call["args"]
            for call in calls
            if call["name"] == "python3" and "-c" not in call["args"]
        ]
        self.assertEqual([args[1] for args in runner_calls], ["check", "list", "array", "array"])
        for args in runner_calls:
            self.assertEqual(args[args.index("--group") + 1], "four-block")

    def test_encoder_filters_and_custom_concurrency_are_forwarded(self):
        result, calls, _, _ = self.launch(
            env_overrides={
                "CP_GROUP": "four-block",
                "CP_ENCODERS": "MAE DINOv3-L",
                "CP_CONCURRENCY": "7",
            },
            args=("--time=48:00:00",),
        )
        self.assert_success(result)
        for call in calls:
            args = call["args"]
            if call["name"] == "python3" and "-c" not in args:
                self.assertEqual(args[args.index("--encoder") + 1:], ["MAE", "DINOv3-L"])
        for submission in self.submissions(calls):
            self.assertIn("--time=48:00:00", submission)
            self.assertTrue(
                next(arg for arg in submission if arg.startswith("--array=")).endswith("%7")
            )

    def test_empty_gpu_groups_are_skipped_without_losing_dependencies(self):
        for group, gpu, empty in (
            ("small", "a100", "v100"),
            ("four-block", "a100", "a100-80gb"),
            ("four-block", "a100-80gb", "a100"),
        ):
            with self.subTest(group=group, gpu=gpu):
                task = "176" if group == "small" else "268" if gpu == "a100-80gb" else "220"
                result, calls, _, _ = self.launch(
                    env_overrides={"CP_GROUP": group},
                    tasks={group: {gpu: task, empty: ""}},
                )
                self.assert_success(result)
                submissions = self.submissions(calls)
                self.assertEqual(len(submissions), 1)
                self.assertIn(f"--array={task}%10", submissions[0])
                self.assertIn("--dependency=afterany:6001:6002", submissions[0])

    def test_no_existing_jobs_has_no_first_dependency(self):
        result, calls, _, _ = self.launch(env_overrides={"EXISTING_JOBS": ""})
        self.assert_success(result)
        submissions = self.submissions(calls)
        self.assertFalse(any(arg.startswith("--dependency=") for arg in submissions[0]))
        self.assertIn("--dependency=afterany:7001", submissions[1])

    def test_runtime_preflight_uses_single_threads_and_environment_libraries(self):
        result, calls, root, _ = self.launch()
        self.assert_success(result)
        preflight = next(
            call for call in calls if call["name"] == "python3" and "-c" in call["args"]
        )
        for name in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"
        ):
            self.assertEqual(preflight["environment"][name], "1")
        self.assertEqual(
            preflight["environment"]["LD_LIBRARY_PATH"], f"{root}/env/lib:/cluster/lib:/cuda/lib"
        )
        self.assertIn("import sqlite3", preflight["args"][-1])
        self.assertIn("RegistryLogger", preflight["args"][-1])

    def test_invalid_group_and_concurrency_are_rejected(self):
        for overrides in (
            {"CP_GROUP": "invalid"}, {"CP_CONCURRENCY": "0"}, {"CP_CONCURRENCY": "11"}
        ):
            with self.subTest(overrides=overrides):
                result, calls, _, _ = self.launch(env_overrides=overrides)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.submissions(calls))

    def test_dependency_overrides_remain_rejected(self):
        for args, overrides in (
            (("--dependency=afterok:9",), {}),
            (("-d", "afterok:9"), {}),
            ((), {"SBATCH_DEPENDENCY": "afterok:9"}),
        ):
            with self.subTest(args=args, overrides=overrides):
                result, calls, _, _ = self.launch(args=args, env_overrides=overrides)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.submissions(calls))


if __name__ == "__main__":
    unittest.main()
