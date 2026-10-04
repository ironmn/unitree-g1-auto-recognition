from fractions import Fraction
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import av
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.observation_collector import IndexedDecoder, PairBuffer, DatasetWriter


def state(index):
    return {'simulate_index': index, 'state': [0.1], 'joint_velocity': [0.2]}


class PairingTests(unittest.TestCase):
    def test_delayed_and_out_of_order_cameras_never_cross_pair(self):
        pairs = PairBuffer(['head', 'wrist_r'])
        for index in [10, 11]:
            pairs.submit(state(index), f'target{index}')
        frame = np.zeros((16, 16, 3), dtype=np.uint8)
        pairs.feed('head', frame, {'simulate_index': 10})
        pairs.feed('wrist_r', frame, {'simulate_index': 11})
        self.assertEqual(pairs.poll(), [])
        pairs.feed('head', frame, {'simulate_index': 11})
        self.assertEqual(pairs.poll(), [])  # keep submission order
        pairs.feed('wrist_r', frame, {'simulate_index': 10})
        ready = pairs.poll()
        self.assertEqual([e['state']['simulate_index'] for e in ready], [10, 11])
        for e in ready:
            self.assertTrue(all(m['simulate_index'] == e['state']['simulate_index']
                                for _, m in e['images'].values()))

    def test_missing_frames_expire_and_memory_is_bounded(self):
        pairs = PairBuffer(['head', 'wrist_r'], timeout=1, capacity=2)
        for i in range(3):
            pairs.submit(state(i), '按钮4')
        self.assertEqual(pairs.dropped, 1)
        self.assertEqual(len(pairs.pending), 2)
        future = max(e['created'] for e in pairs.pending.values()) + 2
        self.assertEqual(pairs.poll(now=future), [])
        self.assertEqual(pairs.dropped, 3)
        with self.assertRaises(ValueError):
            pairs.submit(state(2), 'duplicate')

    def test_sample_files_roundtrip_and_reject_mismatch(self):
        with tempfile.TemporaryDirectory(prefix='观测_') as tmp:
            writer = DatasetWriter(tmp, {'state_names': ['a']})
            image = np.full((16, 32, 3), [20, 40, 200], dtype=np.uint8)
            entry = {'state': state(10), 'prompt': '识别按钮4',
                     'images': {'head': (image, {'simulate_index': 10})}}
            folder = writer.save(entry)
            row = json.loads((folder / 'observation.json').read_text(encoding='utf-8'))
            self.assertEqual(row['prompt'], '识别按钮4')
            self.assertEqual(row['images']['head']['shape_hwc'], [16, 32, 3])
            import cv2
            decoded = cv2.imdecode(np.fromfile(str(folder / 'head.png'), dtype=np.uint8), cv2.IMREAD_COLOR)
            np.testing.assert_array_equal(decoded, image)
            entry['images']['head'] = (image, {'simulate_index': 11})
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                writer.save(entry)
            self.assertEqual(writer.count, 1)
            self.assertEqual(len(list((writer.folder / 'samples').glob('[0-9]*'))), 1)


class DecoderTests(unittest.TestCase):
    def test_pts_tracks_real_h264_reordered_pictures(self):
        encoder = av.CodecContext.create('libx264', 'w')
        encoder.width = encoder.height = 64
        encoder.pix_fmt = 'yuv420p'
        encoder.time_base = Fraction(1, 30)
        encoder.framerate = Fraction(30, 1)
        encoder.options = {'preset': 'medium', 'crf': '10', 'bf': '2', 'g': '30'}
        packets = []
        for i in range(12):
            frame = av.VideoFrame.from_ndarray(np.full((64, 64, 3), 20 + i * 15, dtype=np.uint8), format='bgr24')
            frame.pts = i
            packets.extend(encoder.encode(frame))
        packets.extend(encoder.encode(None))
        self.assertTrue(any(p.pts != p.dts for p in packets))
        decoder = IndexedDecoder()
        results = []
        for packet in packets:
            results.extend(decoder.decode(struct.pack('<Qi', 123, 100 + packet.pts) + bytes(packet)))
        self.assertGreaterEqual(len(results), 8)
        for image, meta in results:
            expected = 20 + (meta['simulate_index'] - 100) * 15
            self.assertLess(abs(float(image.mean()) - expected), 5)

    def test_legacy_or_corrupt_framing_rejected(self):
        decoder = IndexedDecoder()
        with self.assertRaises(ValueError):
            decoder.decode(struct.pack('<Q', 100) + b'\x00\x00\x00\x01\x67' * 4)
        with self.assertRaises(ValueError):
            decoder.decode(b'not a frame')


if __name__ == '__main__':
    unittest.main()
