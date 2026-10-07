import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from g1_openpi_eval import (
    action_metrics,
    paired_comparison,
    stable_window_id,
    summarize_episodes,
    window_starts,
)


def pose(n=24):
    values = np.zeros((n, 18))
    values[:, [6, 13]] = 1
    return values


class EvaluationTests(unittest.TestCase):
    def test_real_field_offsets_units_and_four_grippers(self):
        true = pose()
        pred = true.copy()
        pred[:, 0] = 0.003
        pred[:, 8] = 0.004
        pred[:, 14:18] = [0.1, 0.2, 0.3, 0.4]
        result = action_metrics(pred, true)
        self.assertAlmostEqual(result["left_position_mm_horizon"], 3)
        self.assertAlmostEqual(result["right_position_mm_horizon"], 4)
        self.assertAlmostEqual(result["position_mm_first"], 3.5)
        self.assertAlmostEqual(result["gripper_mae_horizon"], 0.25)

    def test_quaternion_sign_scale_invariance_and_known_rotation(self):
        true = pose()
        pred = -true
        pred[:, 3:7] *= 3
        result = action_metrics(pred, true)
        self.assertAlmostEqual(result["orientation_deg_horizon"], 0)
        self.assertAlmostEqual(result["quaternion_norm_mae"], 1)
        pred = true.copy()
        pred[:, 3:7] = [0, 0, 1, 0]
        result = action_metrics(pred, true)
        self.assertAlmostEqual(result["left_orientation_deg_horizon"], 180)
        self.assertAlmostEqual(result["orientation_deg_horizon"], 90)

    def test_invalid_prediction_penalized_labels_rejected(self):
        true = pose()
        pred = true.copy()
        pred[:, 3:7] = 0
        result = action_metrics(pred, true)
        self.assertAlmostEqual(result["quaternion_invalid_fraction"], 0.5)
        self.assertAlmostEqual(result["left_orientation_deg_horizon"], 180)
        with self.assertRaisesRegex(ValueError, "ground-truth"):
            action_metrics(true, pred)
        pred[0, 0] = np.nan
        with self.assertRaises(ValueError):
            action_metrics(pred, true)
        with self.assertRaises(ValueError):
            action_metrics(np.zeros((24, 32)), np.zeros((24, 32)))

    def test_first_and_horizon_metrics_are_distinct_no_clipping(self):
        true, pred = pose(), pose()
        pred[6:, 0] = 0.1
        pred[:, 14] = 2
        result = action_metrics(pred, true)
        self.assertEqual(result["position_mm_first6"], 0)
        self.assertAlmostEqual(result["position_mm_horizon"], 37.5)
        self.assertAlmostEqual(result["gripper_mae_first"], 0.5)
        self.assertEqual(result["gripper_out_of_range_fraction"], 0.25)

    def test_unique_windows_include_final_without_padding(self):
        self.assertEqual(window_starts(49, 24, 24), [0, 24, 25])
        self.assertEqual(window_starts(48, 24, 24), [0, 24])
        self.assertEqual(window_starts(24, 24, 1), [0])
        with self.assertRaises(ValueError):
            window_starts(23, 24, 24)
        with self.assertRaises(ValueError):
            window_starts(48, 24, 0)

    def test_rng_is_stable_and_has_distinct_namespaces(self):
        a = stable_window_id("episodeA", 3, "loss")
        self.assertEqual(a, stable_window_id("episodeA", 3, "loss"))
        self.assertNotEqual(a, stable_window_id("episodeA", 3, "action"))
        self.assertNotEqual(a, stable_window_id("episodeB", 3, "loss"))

    def test_episode_and_task_macro_weight_not_window_count(self):
        episodes = []
        for key, task, value, windows in [
            ("a", "press", 0, 1000),
            ("b", "press", 10, 1),
            ("c", "rotate", 20, 100),
        ]:
            episodes.append(
                {
                    "key": key,
                    "task": task,
                    "loss_windows": windows,
                    "action_windows": 1,
                    "metrics": {"error": value},
                    "hold_current_state": {"error": value},
                }
            )
        result = summarize_episodes(episodes)
        self.assertEqual(result["tasks"]["press"]["metrics"]["error"], 5)
        self.assertEqual(result["macro"]["error"], 12.5)
        a = {"plan_sha256": "same", "episodes": episodes, "summary": result}
        b = {**a, "plan_sha256": "different"}
        with self.assertRaisesRegex(ValueError, "plans"):
            paired_comparison(a, b)
        self.assertEqual(paired_comparison(a, a)["macro"]["error"]["delta"], 0)


if __name__ == "__main__":
    unittest.main()
