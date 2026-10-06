"""Contract regressions for G1/OpenPI temporal and camera semantics."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from g1_openpi_data import CAMERAS, G1Dataset, G1Inputs, G1Outputs


class G1ContractTests(unittest.TestCase):
    def observation(self):
        return {
            "observation.state": np.zeros(18, np.float32),
            "prompt": "按压停止按钮",
            "observation.images.cam_head": np.full((3, 20, 30), 0.5, np.float32),
            "observation.images.cam_wrist_r": np.full((20, 30, 3), 255, np.uint8),
        }

    def test_actual_right_camera_and_missing_left_mask(self):
        out = G1Inputs()(self.observation())
        self.assertTrue(out["image_mask"]["right_wrist_0_rgb"])
        self.assertFalse(out["image_mask"]["left_wrist_0_rgb"])
        self.assertEqual(out["image"]["right_wrist_0_rgb"][112, 112, 0], 255)
        self.assertEqual(out["image"]["base_0_rgb"][112, 112, 0], 128)
        self.assertNotIn("actions", out)

    def test_future_window_does_not_shift_labels_again(self):
        ds = G1Dataset.__new__(G1Dataset)
        ds.horizon = 2
        ds.tasks = {0: "press"}
        ds.indices = [(0, 0), (0, 1), (1, 0), (1, 1)]
        ds.episodes = []
        for offset in [0, 100]:
            ds.episodes.append(
                {
                    "states": np.full((3, 18), offset),
                    "actions": np.repeat((np.arange(3) + offset + 1)[:, None], 18, axis=1),
                    "tasks": [0] * 3,
                    "images": {k: np.zeros((3, 4, 4, 3), np.uint8) for k in CAMERAS.values()},
                }
            )
        np.testing.assert_array_equal(ds[1]["action"][:, 0], [2, 3])
        np.testing.assert_array_equal(ds[2]["action"][:, 0], [101, 102])

    def test_wrong_action_semantics_rejected_by_shape(self):
        obs = self.observation()
        obs["action"] = np.zeros((24, 45))
        with self.assertRaises(ValueError):
            G1Inputs()(obs)

    def test_output_crops_padding_and_rejects_nonfinite(self):
        actions = np.ones((2, 24, 32))
        self.assertEqual(G1Outputs()({"actions": actions})["actions"].shape, (2, 24, 18))
        actions[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            G1Outputs()({"actions": actions})


if __name__ == "__main__":
    unittest.main()
