"""Real local WebSocket -> H.264 -> bounded indexed-frame queue."""

import asyncio
import struct
import time
import unittest

import av
import numpy as np
import websockets

from unitree_vision.streams import IndexedCameraStream


class StreamIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_receiver_preserves_index_and_counts_overflow(self):
        encoder = av.CodecContext.create("libx264", "w")
        encoder.width = encoder.height = 64
        encoder.pix_fmt = "yuv420p"
        encoder.options = {"preset": "ultrafast", "tune": "zerolatency"}
        messages = []
        for index in range(3):
            frame = av.VideoFrame.from_ndarray(
                np.full((64, 64, 3), 30 + index * 30, dtype=np.uint8), format="bgr24"
            )
            packets = encoder.encode(frame)
            self.assertEqual(len(packets), 1)
            messages.append(struct.pack("<Qi", 100 + index, 10 + index) + bytes(packets[0]))

        async def handler(socket):
            for message in messages:
                await socket.send(message)
            await socket.wait_closed()

        async with websockets.serve(handler, "127.0.0.1", 0) as server:
            stream = IndexedCameraStream(
                "head", "127.0.0.1", server.sockets[0].getsockname()[1], capacity=2
            )
            stream.thread.start()
            try:
                deadline = time.monotonic() + 5
                while stream.snapshot()[1] < 3 and time.monotonic() < deadline:
                    await asyncio.sleep(0.01)
                self.assertEqual(stream.snapshot()[1], 3, stream.snapshot()[4])
                frames = stream.drain()
                self.assertEqual([meta["simulate_index"] for _, meta in frames], [11, 12])
                self.assertEqual(stream.overflow, 1)
                self.assertEqual(stream.drain(), [])
            finally:
                await asyncio.to_thread(stream.stop)
            self.assertFalse(stream.thread.is_alive())
