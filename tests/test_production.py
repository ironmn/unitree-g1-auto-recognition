"""Failure-mode regression tests independent of GPU, OrcaLab and robot hardware."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from unitree_vision.collector import ObservationCollector, parser
from unitree_vision.config import CollectorConfig, parse_config
from unitree_vision.dataset import DatasetWriter, inspect_dataset, rebuild_index
from unitree_vision.pairing import PairBuffer


def manifest():
    return {
        "schema_version": 1,
        "state_manifest": {"state_names": ["joint"], "schema_id": "test"},
        "cameras": {"head": {"width": 32, "height": 16}},
    }


def entry(index=10):
    return {
        "state": {
            "simulate_index": index,
            "schema_id": "test",
            "state": [0.1],
            "joint_velocity": [0.2],
        },
        "prompt": "识别按钮4",
        "images": {
            "head": (np.full((16, 32, 3), [10, 30, 200], dtype=np.uint8), {"simulate_index": index})
        },
    }


class ConfigurationTests(unittest.TestCase):
    def test_invalid_api_settings_fail_before_connecting(self):
        for settings in (
            {"fps": float("nan")},
            {"pair_capacity": 0},
            {"cameras": "head,head"},
            {"profile": "invalid"},
            {"camera_buffer": True},
            {"record": "false"},
            {"output": 3},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                CollectorConfig(**settings)

    def test_cli_defaults_and_overrides(self):
        self.assertEqual(parse_config(parser(), []).fps, 10)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "collector.toml"
            path.write_text(
                '[collector]\nfps=5\nrecord=true\nheadless=true\nprompt="按钮4"\n', encoding="utf-8"
            )
            config = parse_config(parser(), ["--config", str(path), "--fps", "12", "--no-headless"])
            self.assertEqual(config.fps, 12)
            self.assertTrue(config.record)
            self.assertFalse(config.headless)
            self.assertEqual(config.prompt, "按钮4")
            path.write_text("[collector]\nunknown=true\n")
            with self.assertRaises(SystemExit):
                parse_config(parser(), ["--config", str(path)])

    def test_headless_without_record_rejected(self):
        with self.assertRaises(SystemExit):
            parse_config(parser(), ["--headless"])


class TransactionTests(unittest.TestCase):
    def test_failed_encode_cleans_partial_and_same_index_can_retry(self):
        with tempfile.TemporaryDirectory(prefix="观测_") as tmp:
            writer = DatasetWriter(tmp, manifest())
            with patch("unitree_vision.dataset.cv2.imencode", return_value=(False, None)):
                with self.assertRaises(RuntimeError):
                    writer.save(entry())
            self.assertEqual(list((writer.folder / "samples").iterdir()), [])
            self.assertEqual(writer.count, 0)
            writer.save(entry())
            self.assertEqual(inspect_dataset(writer.folder)["samples"], 1)

    def test_index_failure_keeps_committed_sample_and_rebuilds(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = DatasetWriter(tmp, manifest())
            original = Path.open

            def failing_open(path, *args, **kwargs):
                if path.name == "index.jsonl" and args and args[0] == "a":
                    raise OSError("simulated disk error")
                return original(path, *args, **kwargs)

            with patch.object(Path, "open", failing_open):
                with self.assertRaises(OSError):
                    writer.save(entry())
            self.assertEqual(writer.count, 1)
            self.assertEqual(len(list((writer.folder / "samples").iterdir())), 1)
            with self.assertRaisesRegex(ValueError, "index.jsonl"):
                inspect_dataset(writer.folder)
            rebuilt = rebuild_index(writer.folder)
            self.assertEqual(rebuilt["samples"], 1)
            self.assertEqual(inspect_dataset(writer.folder)["images"], 1)
            self.assertEqual(len(list(writer.folder.glob("index.jsonl.backup_*"))), 1)

    def test_alignment_schema_and_corruption_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = DatasetWriter(tmp, manifest())
            sample = writer.save(entry())
            path = sample / "observation.json"
            original = json.loads(path.read_text(encoding="utf-8"))
            for bad in ("schema", "index", "dimensions", "path"):
                record = json.loads(json.dumps(original))
                if bad == "schema":
                    record["observation"]["schema_id"] = "wrong"
                if bad == "index":
                    record["images"]["head"]["simulate_index"] = 11
                if bad == "dimensions":
                    record["observation"]["state"] = [1.0, 2.0]
                if bad == "path":
                    record["images"]["head"]["file"] = "../outside.png"
                path.write_text(json.dumps(record), encoding="utf-8")
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    inspect_dataset(writer.folder)
            path.write_text(json.dumps(original), encoding="utf-8")
            (sample / "head.png").write_bytes(b"bad PNG")
            with self.assertRaisesRegex(ValueError, "corrupt"):
                inspect_dataset(writer.folder)

    def test_partial_directories_are_excluded_and_not_deleted(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = DatasetWriter(tmp, manifest())
            writer.save(entry())
            partial = writer.folder / "samples/.partial_interrupted"
            partial.mkdir()
            result = rebuild_index(writer.folder)
            self.assertEqual(result["samples"], 1)
            self.assertEqual(result["partial_samples"], 1)
            self.assertTrue(partial.is_dir())

    def test_writer_rejects_incomplete_or_invalid_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = DatasetWriter(tmp, manifest())
            missing = entry()
            missing["images"] = {}
            with self.assertRaises(ValueError):
                writer.save(missing)
            invalid = entry()
            invalid["images"]["head"] = (np.zeros((2, 2)), {"simulate_index": 10})
            with self.assertRaises(ValueError):
                writer.save(invalid)


class PairingBoundaryTests(unittest.TestCase):
    def test_protocol_bounds_and_state_snapshot(self):
        pairs = PairBuffer(["head"])
        for index in (True, -1, 2**31):
            with self.assertRaises(ValueError):
                pairs.submit({"simulate_index": index}, "prompt")
        sample = entry()
        pairs.submit(sample["state"], sample["prompt"])
        sample["state"]["state"][0] = 999
        frame, meta = sample["images"]["head"]
        pairs.feed("head", frame, meta)
        self.assertEqual(pairs.poll()[0]["state"]["state"], [0.1])


class LifecycleTests(unittest.TestCase):
    def make(self, tmp):
        reader = SimpleNamespace(
            manifest={"state_names": ["joint"], "schema_id": "test"},
            read=lambda **kw: entry(kw["simulate_index"])["state"],
        )
        with patch("unitree_vision.collector.StateReader.from_env", return_value=reader):
            return ObservationCollector(object(), CollectorConfig(cameras="head", output=Path(tmp)))

    def test_summary_failure_always_closes_owned_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            collector = self.make(tmp)
            collector.session = SimpleNamespace(streams={}, close=unittest.mock.Mock())
            collector.writer = DatasetWriter(tmp, manifest())
            with patch("unitree_vision.collector.atomic_json", side_effect=OSError("disk full")):
                with self.assertLogs("unitree_vision.collector", level="ERROR"):
                    collector.close()
            collector.close()
            collector.session.close.assert_called_once()
            with self.assertRaises(RuntimeError):
                collector.submit(1)

    def test_open_failure_closes_resources(self):
        with tempfile.TemporaryDirectory() as tmp:
            collector = self.make(tmp)
            fake_session = SimpleNamespace(
                open=unittest.mock.Mock(side_effect=RuntimeError("camera failure")),
                close=unittest.mock.Mock(),
            )
            with patch("unitree_vision.collector.PreviewSession", return_value=fake_session):
                with self.assertRaisesRegex(RuntimeError, "camera failure"):
                    collector.open()
            fake_session.close.assert_called_once()

    def test_embedded_open_uses_receive_only_and_records_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            collector = self.make(tmp)
            stream = SimpleNamespace(drain=lambda: [], overflow=0)
            session = SimpleNamespace(
                streams={"head": stream}, open=unittest.mock.Mock(), close=unittest.mock.Mock()
            )

            def call(method, request):
                if method == "GetCameraNames":
                    return SimpleNamespace(status=0, camera_names=["head_cam_uuid"])
                return SimpleNamespace(status=0, width=32, height=16, color_port=7090)

            session.call = call
            with patch("unitree_vision.collector.PreviewSession", return_value=session) as factory:
                with collector:
                    self.assertTrue(factory.call_args.args[0].receive_only)
                    value = json.loads(
                        (collector.writer.folder / "manifest.json").read_text(encoding="utf-8")
                    )
                    self.assertEqual(value["mode"], "controller_integration")
                    self.assertFalse(value["physics_stepped_by_collector"])
                    self.assertEqual(value["cameras"]["head"]["name"], "head_cam_uuid")
                    self.assertEqual(value["resolved_config"]["cameras"], "head")
            session.close.assert_called_once()

    def test_poll_sampling_and_save_next(self):
        with tempfile.TemporaryDirectory() as tmp:
            collector = self.make(tmp)
            collector.args = replace(collector.args, record=True, sample_every=2)
            collector.recording = True
            stream = SimpleNamespace(drain=lambda: [], overflow=0)
            collector.session = SimpleNamespace(streams={"head": stream}, close=lambda: None)
            collector.writer = DatasetWriter(tmp, manifest())
            for index in range(4):
                sample = entry(index)
                collector.submit(index)
                collector.pairs.feed("head", *sample["images"]["head"])
                collector.poll()
            self.assertEqual(collector.writer.count, 2)
            collector.recording = False
            collector.save_next = True
            collector.submit(4)
            collector.pairs.feed("head", *entry(4)["images"]["head"])
            collector.poll()
            self.assertEqual(collector.writer.count, 3)
            self.assertFalse(collector.save_next)
            self.assertEqual(inspect_dataset(collector.writer.folder)["samples"], 3)
            collector.close()


if __name__ == "__main__":
    unittest.main()
