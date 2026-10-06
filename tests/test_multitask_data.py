import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from g1_openpi_data import G1MultiTaskDataset


class MultiTaskTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.spec = {"schema_version": 1, "datasets": []}
        for name in ["press", "rotate", "toggle"]:
            (self.root / f"{name}.json").write_text(
                json.dumps(
                    {
                        "evaluation": {"proxy_success": True, "official_success": None},
                        "goal": {"target_id": name},
                    }
                )
            )
            self.spec["datasets"].append(
                {
                    "name": name,
                    "path": name,
                    "expected_prompt": name,
                    "target_id": name,
                    "evaluation": f"{name}.json",
                }
            )
        self.path = self.root / "manifest.json"
        self.path.write_text(json.dumps(self.spec))

    def tearDown(self):
        self.temp.cleanup()

    def fake(self, path, horizon):
        name = path.name
        n = {"press": 25, "rotate": 28, "toggle": 26}[name]
        i = {"press": 0, "rotate": 1, "toggle": 2}[name]
        return SimpleNamespace(
            tasks={0: name},
            info={"total_frames": n},
            indices=[(0, t) for t in range(n - horizon + 1)],
            episodes=[
                {
                    "states": np.full((n, 18), i),
                    "actions": np.full((n, 18), i),
                    "images": {},
                    "tasks": [0] * n,
                }
            ],
        )

    def test_balanced_tasks_and_no_cross_episode(self):
        with patch("g1_openpi_data.G1Dataset", side_effect=self.fake):
            ds = G1MultiTaskDataset(self.path)
        self.assertEqual(
            Counter(ds[i]["prompt"] for i in range(len(ds))), {"press": 5, "rotate": 5, "toggle": 5}
        )
        for i in range(len(ds)):
            e, t = ds.indices[i]
            self.assertLessEqual(t + 24, len(ds.episodes[e]["states"]))
            self.assertTrue(np.all(ds[i]["action"] == e))

    def test_failed_quality_gate_rejected(self):
        (self.root / "press.json").write_text(
            json.dumps({"evaluation": {"proxy_success": False}, "goal": {"target_id": "press"}})
        )
        with patch("g1_openpi_data.G1Dataset", side_effect=self.fake):
            with self.assertRaises(ValueError):
                G1MultiTaskDataset(self.path)

    def test_task_relabel_rejected(self):
        self.spec["datasets"][0]["expected_prompt"] = "different"
        self.path.write_text(json.dumps(self.spec))
        with patch("g1_openpi_data.G1Dataset", side_effect=self.fake):
            with self.assertRaises(ValueError):
                G1MultiTaskDataset(self.path)

    def test_multiple_sources_same_task_balance_by_task(self):
        extra = dict(self.spec["datasets"][0], path="extra/press")
        self.spec["datasets"].append(extra)
        self.path.write_text(json.dumps(self.spec))
        with patch("g1_openpi_data.G1Dataset", side_effect=self.fake):
            ds = G1MultiTaskDataset(self.path)
        self.assertEqual(len(ds.tasks), 3)
        self.assertEqual(len(ds.episodes), 4)
        self.assertEqual(
            Counter(ds[i]["prompt"] for i in range(len(ds))), {"press": 5, "rotate": 5, "toggle": 5}
        )
        self.assertIn(3, {e for e, t in ds.indices})


if __name__ == "__main__":
    unittest.main()
