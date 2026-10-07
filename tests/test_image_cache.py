import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from g1_openpi_data import read_video_images


class ImageCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.video = self.root / "camera.mp4"
        self.video.write_bytes(b"source video version one")
        self.cache = self.root / "cache"
        self.rgb = [np.full((24, 32, 3), (i * 40, 100, 200), np.uint8) for i in range(3)]

    def tearDown(self):
        self.temp.cleanup()

    @contextmanager
    def decoder(self, _path):
        rgb = self.rgb

        class Frame:
            def __init__(self, image):
                self.image = image

            def to_ndarray(self, format):
                assert format == "rgb24"
                return self.image

        class Container:
            def decode(self, video):
                assert video == 0
                return (Frame(im) for im in rgb)

        yield Container()

    def test_cache_is_pixel_identical_readonly_and_reused(self):
        with patch("g1_openpi_data.av.open", side_effect=self.decoder):
            original = read_video_images(self.video, 3)
            cached = read_video_images(self.video, 3, self.cache)
        np.testing.assert_array_equal(original, cached)
        self.assertIsInstance(cached, np.memmap)
        self.assertFalse(cached.flags.writeable)
        with patch(
            "g1_openpi_data.av.open", side_effect=AssertionError("Cache must avoid decoding")
        ):
            reused = read_video_images(self.video, 3, self.cache)
        np.testing.assert_array_equal(original, reused)
        self.assertEqual(original.shape, (3, 224, 224, 3))

    def test_source_changes_invalidate_cache(self):
        with patch("g1_openpi_data.av.open", side_effect=self.decoder):
            before = read_video_images(self.video, 3, self.cache).copy()
            self.video.write_bytes(b"different source video content")
            self.rgb = [np.zeros_like(im) for im in self.rgb]
            after = read_video_images(self.video, 3, self.cache)
        self.assertFalse(np.array_equal(before, after))
        self.assertEqual(len(list(self.cache.glob("*.npy"))), 2)

    def test_wrong_frame_count_does_not_publish_cache(self):
        with patch("g1_openpi_data.av.open", side_effect=self.decoder):
            for n in [2, 4]:
                with self.assertRaisesRegex(ValueError, "Video/table mismatch"):
                    read_video_images(self.video, n, self.cache)
        self.assertEqual(list(self.cache.glob("*.npy")), [])

    def test_invalid_cached_shape_is_rejected(self):
        with patch("g1_openpi_data.av.open", side_effect=self.decoder):
            read_video_images(self.video, 3, self.cache)
        target = next(self.cache.glob("*.npy"))
        np.save(target, np.zeros((2, 224, 224, 3), np.uint8))
        with self.assertRaisesRegex(ValueError, "Invalid image cache"):
            read_video_images(self.video, 3, self.cache)


if __name__ == "__main__":
    unittest.main()
