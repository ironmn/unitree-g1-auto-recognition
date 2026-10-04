"""Bounded indexed camera receivers."""

import asyncio
import time
from collections import deque

import websockets

from .camera import CameraStream
from .protocol import IndexedDecoder


class IndexedCameraStream(CameraStream):
    def __init__(self, role, host, port, *, capacity=16):
        super().__init__(role, host, port)
        if type(capacity) is not int or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        self.frames = deque(maxlen=capacity)
        self.connected = False
        self.overflow = 0

    async def _receive(self):
        while not self.stop_event.is_set():
            try:
                decoder = IndexedDecoder()
                with self.lock:
                    self.frames.clear()
                    self.frame = None
                    self.metadata = {}

                async with websockets.connect(
                    self.uri,
                    open_timeout=3,
                    close_timeout=1,
                    max_size=16 * 1024 * 1024,
                    max_queue=2,
                ) as ws:
                    with self.lock:
                        self.connected = True
                        self.error = "connected; waiting for indexed keyframe"
                    while not self.stop_event.is_set():
                        try:
                            message = await asyncio.wait_for(ws.recv(), timeout=0.5)
                        except TimeoutError:
                            continue
                        for frame, meta in decoder.decode(message):
                            with self.lock:
                                if len(self.frames) == self.frames.maxlen:
                                    self.overflow += 1
                                self.frames.append((frame, meta))
                                self.frame = frame
                                self.metadata = meta
                                self.index += 1
                                self.frame_times.append(time.monotonic())
                                self.error = ""
            except Exception as exc:
                with self.lock:
                    self.connected = False
                    self.error = f"{type(exc).__name__}: {exc}"
                await asyncio.sleep(0.5)
            finally:
                with self.lock:
                    self.connected = False

    def drain(self):
        with self.lock:
            frames = list(self.frames)
            self.frames.clear()
            return frames
