"""Frozen-evaluation seed recovery must never impute unobserved runs."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from eval.utils import final_integration as integration


class SeedAggregationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        self.archive = pd.DataFrame([dict(
            Method="DIET-CP", Backbone="MAE", dataset_key="octmnist", size=1000,
            knn_pre=0.2, lp_pre=0.3, knn_post=0.9, lp_post=0.8,
            knn_post_s=0.01, lp_post_s=0.02, dknn=0.7, dlp=0.5,
            ft_pre=0.5, ft_post=0.6, dft=0.1,
        )])
        self.reruns = pd.DataFrame([
            dict(method="DIET", encoder="MAE", dataset="octmnist", n=1000,
                 seed=seed, post_knn=knn, post_lp=lp, pre_knn=0.2, pre_lp=0.3,
                 d_knn=round(knn - 0.2, 4), d_lp=round(lp - 0.3, 4),
                 d_ft=0.4, results_json=f"seed{seed}.json")
            for seed, knn, lp in ((43, 0.51234567, 0.62345678), (44, 0.43123456, 0.73234567))
        ])
        self.recovered = pd.DataFrame([dict(
            method="DIET", encoder="MAE", dataset="octmnist", n=1000, seed=42,
            post_knn=0.32123456, post_lp=0.51234567,
            source="author_report_20260921", seed_assignment="unique_missing_seed",
        )])

    def refresh(self):
        self.archive.to_csv(self.out / "cp_long.csv", index=False)
        self.archive.to_csv(self.out / "cp_long_refreshed.csv", index=False)
        self.reruns.to_csv(self.out / "rest_behavior.csv", index=False)
        self.recovered.to_csv(self.out / "main_grid_recovered_seeds_20260921.csv", index=False)
        with patch.object(integration, "OUT", self.out), redirect_stdout(io.StringIO()):
            return integration.refresh_cp_long()

    def test_true_three_seed_means_effects_and_sample_sd_are_synchronized(self):
        result = self.refresh().iloc[0]
        for metric, baseline, raw in (
            ("knn", 0.2, [0.32123456, 0.51234567, 0.43123456]),
            ("lp", 0.3, [0.51234567, 0.62345678, 0.73234567]),
        ):
            self.assertAlmostEqual(result[f"{metric}_post"], np.mean(raw), places=14)
            self.assertAlmostEqual(result[f"{metric}_post_s"], np.std(raw, ddof=1), places=14)
            self.assertAlmostEqual(result[f"d{metric}"], np.mean(raw) - baseline, places=14)

    def test_unrelated_ft_fields_are_not_refreshed(self):
        result = self.refresh()
        columns = ["ft_pre", "ft_post", "dft"]
        pd.testing.assert_frame_equal(result[columns], self.archive[columns])

    def test_missing_seed_is_rejected_without_writing_an_approximate_result(self):
        self.recovered = self.recovered.iloc[:0]
        with self.assertRaisesRegex(ValueError, "three seeds|42, 43, 44"):
            self.refresh()
        pd.testing.assert_frame_equal(pd.read_csv(self.out / "cp_long_refreshed.csv"), self.archive)

    def test_duplicate_seed_is_rejected(self):
        self.recovered = pd.concat([self.recovered, self.recovered], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.refresh()

    def test_nonfinite_score_is_rejected(self):
        self.recovered.loc[0, "post_lp"] = np.nan
        with self.assertRaisesRegex(ValueError, "finite|macro-F1"):
            self.refresh()

    def test_wrong_baseline_reference_is_rejected(self):
        self.reruns.loc[0, "pre_knn"] = 0.3
        with self.assertRaisesRegex(ValueError, "baseline"):
            self.refresh()

    def test_unmatched_configuration_is_rejected(self):
        self.recovered.loc[0, "n"] = 500
        with self.assertRaisesRegex(ValueError, "configuration|three seeds|42, 43, 44"):
            self.refresh()

    def test_inferred_seed_requires_one_missing_seed(self):
        self.reruns = self.reruns.iloc[:1]
        with self.assertRaisesRegex(ValueError, "missing seed|three seeds|42, 43, 44"):
            self.refresh()


if __name__ == "__main__":
    unittest.main()
