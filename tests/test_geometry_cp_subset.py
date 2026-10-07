"""Preserve budget-specific geometry when kNN uses the entire training split."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np

from test_lp_full_train import ROOT, load_definitions


class GeometrySubsetTests(unittest.TestCase):
    def setUp(self):
        self.descriptors = Mock(side_effect=lambda bank, ref: {"n_geometry": len(bank)})
        self.namespace = {
            "np": np, "json": json, "Path": Path, "tempfile": tempfile,
            "_BANK_SIZE": 5000, "GEOMETRY_PROTOCOL": "precp_geometry_5000_v1",
            "geometry_descriptors": self.descriptors,
        }
        load_definitions(ROOT / "stable_cp/evaluation/geometry.py",
                         {"select_geometry_indices", "evaluate_geometry"}, self.namespace)

    def evaluate(self, features, **kwargs):
        metadata = dict(backbone="encoder", pool_strategy="cls", normalization={})
        with tempfile.TemporaryDirectory() as folder:
            reference = Path(folder) / "reference.npz"
            np.savez(reference, features=np.ones((5000, 2)), metadata=json.dumps(dict(
                protocol="precp_geometry_5000_v1", n_reference=5000, **metadata
            )))
            return self.namespace["evaluate_geometry"](
                features, reference, metadata=metadata, **kwargs
            )

    def test_geometry_uses_cp_subset_in_original_order(self):
        features = np.arange(12).reshape(6, 2)
        result = self.evaluate(features, target_indices=[4, 1])
        np.testing.assert_array_equal(self.descriptors.call_args.args[0], features[[4, 1]])
        self.assertEqual(result["n_train"], 2)
        self.assertEqual(result["n_geometry"], 2)

    def test_full_budget_geometry_is_unchanged(self):
        features = np.arange(12000).reshape(6000, 2)
        result = self.evaluate(features)
        indices = self.namespace["select_geometry_indices"](6000)
        np.testing.assert_array_equal(self.descriptors.call_args.args[0], features[indices])
        self.assertEqual(result["n_train"], 6000)
        self.assertEqual(result["n_geometry"], 5000)


if __name__ == "__main__":
    unittest.main()
