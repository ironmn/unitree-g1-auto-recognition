"""Reject malformed transport observations without importing JAX or OpenPI."""

import io
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from serve_g1_openpi import decode_observation


class ObservationTransportTests(unittest.TestCase):
    def payload(self, **changes):
        values = {
            "state": np.zeros(18, np.float32),
            "prompt": np.asarray("按压停止按钮"),
            "seed": np.uint32(42),
            "observation.images.cam_head": np.zeros((24, 32, 3), np.uint8),
            "observation.images.cam_wrist_r": np.ones((24, 32, 3), np.uint8),
        }
        values.update(changes)
        stream = io.BytesIO()
        np.savez_compressed(stream, **values)
        return stream.getvalue()

    def test_rgb_and_prompt_are_preserved(self):
        raw, seed = decode_observation(self.payload())
        self.assertEqual(seed, 42)
        self.assertEqual(raw["prompt"], "按压停止按钮")
        self.assertEqual(raw["observation.state"].shape, (18,))
        self.assertEqual(raw["observation.images.cam_wrist_r"].dtype, np.uint8)

    def test_invalid_contracts_rejected(self):
        cases = [
            {"state": np.zeros(17)},
            {"state": np.full(18, np.nan)},
            {"prompt": np.asarray(["press"])},
            {"prompt": np.asarray("")},
            {"seed": np.int64(-1)},
            {"observation.images.cam_head": np.zeros((24, 32, 3), np.float32)},
            {"extra": np.zeros(1)},
        ]
        for case in cases:
            with self.subTest(fields=list(case)), self.assertRaises(ValueError):
                decode_observation(self.payload(**case))


if __name__ == "__main__":
    unittest.main()
