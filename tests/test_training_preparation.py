import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_multitask_training import training_arrays


class PreparationTests(unittest.TestCase):
    def test_overlap_rejected_before_reading_training_data(self):
        source = [(Path("/same"), {"expected_prompt": "press"})]
        with (
            patch("prepare_multitask_training.sources", return_value=source),
            patch("prepare_multitask_training.pq.read_table") as reader,
        ):
            with self.assertRaisesRegex(ValueError, "leakage"):
                training_arrays(Path("train"), Path("val"))
            reader.assert_not_called()

    def test_validation_values_never_read_for_statistics(self):
        import json
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = root / "train"
            val = root / "validation"
            (train / "data").mkdir(parents=True)
            (train / "meta").mkdir()
            (train / "data/episode.parquet").touch()
            (train / "meta/info.json").write_text(
                json.dumps({"fps": 24, "total_episodes": 1, "total_frames": 24})
            )
            states = np.arange(24 * 18, dtype=np.float32).reshape(24, 18)
            actions = np.concatenate([states[1:], states[-1:]])
            table = {
                "observation.state": states,
                "action": actions,
                "frame_index": list(range(24)),
                "timestamp": np.arange(24) / 24,
            }
            from types import SimpleNamespace

            with (
                patch(
                    "prepare_multitask_training.sources",
                    side_effect=[
                        [(train, {"expected_prompt": "press"})],
                        [(val, {"expected_prompt": "press"})],
                    ],
                ),
                patch(
                    "prepare_multitask_training.pq.read_table",
                    return_value=SimpleNamespace(to_pydict=lambda: table),
                ) as reader,
            ):
                arrays, details, count = training_arrays(
                    Path("train_manifest"), Path("val_manifest")
                )
            self.assertEqual(reader.call_count, 1)
            self.assertEqual(reader.call_args.args[0], train / "data/episode.parquet")
            self.assertTrue(np.array_equal(arrays["state"], states))
            self.assertEqual(count, 1)


if __name__ == "__main__":
    unittest.main()
