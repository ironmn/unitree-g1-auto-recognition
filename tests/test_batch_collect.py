import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from batch_collect import summarize, variant, write


class BatchTests(unittest.TestCase):
    def test_variant_preserves_contact_and_is_reproducible(self):
        template = {
            "segments": [
                {"steps": 500, "r_target_b": [0.2, 0.3, 0.4]},
                {"steps": 1000, "r_target_b": [0.5, 0.6, 0.7], "gripper": 1},
            ]
        }
        original = copy.deepcopy(template)
        a, meta = variant(template, 42)
        self.assertEqual((a, meta), variant(template, 42))
        self.assertEqual(template, original)
        self.assertEqual(a["segments"][1]["r_target_b"], template["segments"][1]["r_target_b"])
        self.assertEqual(a["segments"][1]["gripper"], 1)
        self.assertTrue(0.92 <= meta["duration_scale"] <= 1.08)
        self.assertTrue(
            all(
                abs(v) <= limit
                for v, limit in zip(meta["first_waypoint_offset_m"], [0.008, 0.008, 0.006])
            )
        )

    def test_rejected_and_incomplete_never_enter_disjoint_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = {
                "tasks": {"press": {"prompt": "press", "target": "stop"}},
                "per_task": 2,
                "validation_per_task": 1,
            }
            for n, ok in [(0, False), (1, True), (2, True)]:
                write(
                    root / f"attempts/{n:03d}_press/result.json",
                    {
                        "task": "press",
                        "accepted": ok,
                        "directory": f"attempts/{n:03d}_press",
                        "metrics": {"frames": 100},
                    },
                )
            (root / "attempts/003_press").mkdir()
            report = summarize(root, plan)
            self.assertEqual(
                (report["accepted"], report["rejected"], report["frames"]), (2, 1, 200)
            )
            train = json.loads((root / "manifest_train.json").read_text())["datasets"]
            val = json.loads((root / "manifest_validation.json").read_text())["datasets"]
            self.assertEqual(len(train), 1)
            self.assertEqual(len(val), 1)
            self.assertNotEqual(train[0]["path"], val[0]["path"])


if __name__ == "__main__":
    unittest.main()
