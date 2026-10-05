"""Offline protocol and real H.264 decoder checks; no OrcaLab connection."""

import importlib.util
import struct
import unittest
from pathlib import Path

import av
import numpy as np

spec = importlib.util.spec_from_file_location(
    "preview", Path(__file__).resolve().parents[1] / "scripts" / "camera_preview.py"
)
preview = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preview)


class CameraProtocolTests(unittest.TestCase):
    def test_framing_variants_and_corruption(self):
        nal = b"\x00\x00\x00\x01\x67\x42\x00\x1e"
        for header in (b"", struct.pack("<Q", 123), struct.pack("<Qi", 123, 45)):
            payload, meta = preview.unpack_message(header + nal)
            self.assertEqual(payload, nal)
            self.assertEqual(meta["header_bytes"], len(header))
        with self.assertRaises(ValueError):
            preview.unpack_message(b"invalid framing")

    def test_real_h264_incremental_decode(self):
        encoder = av.CodecContext.create("libx264", "w")
        encoder.width = encoder.height = 64
        encoder.pix_fmt = "yuv420p"
        encoder.options = {"preset": "ultrafast", "tune": "zerolatency", "crf": "18"}
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[:, :, 2] = 220  # red in BGR
        packets = []
        for _ in range(4):
            packets.extend(encoder.encode(av.VideoFrame.from_ndarray(image, format="bgr24")))
        packets.extend(encoder.encode(None))
        decoder = av.CodecContext.create("h264", "r")
        frames = []
        for i, packet in enumerate(packets):
            payload, _ = preview.unpack_message(struct.pack("<Qi", 123 + i, i) + bytes(packet))
            # Exercise fragmented input rather than assume one message = one frame.
            for start in range(0, len(payload), 37):
                for parsed in decoder.parse(payload[start : start + 37]):
                    frames.extend(decoder.decode(parsed))
        for parsed in decoder.parse(b""):
            frames.extend(decoder.decode(parsed))
        frames.extend(decoder.decode(None))
        self.assertEqual(len(frames), 4)
        result = frames[0].to_ndarray(format="bgr24")
        self.assertEqual(result.shape, (64, 64, 3))
        self.assertGreater(result[:, :, 2].mean(), 200)
        self.assertLess(result[:, :, 0].mean(), 10)

    def test_ambiguous_camera_rejected(self):
        with self.assertRaises(ValueError):
            preview.resolve_camera(["head_cam_one", "head_cam_two"], "head")
        self.assertEqual(preview.resolve_camera(["head_cam_uuid"], "head"), "head_cam_uuid")


if __name__ == "__main__":
    unittest.main()
