"""Exact camera packet/frame metadata association."""

import struct
import time
from collections import OrderedDict
from fractions import Fraction

import av


class IndexedDecoder:
    """Engine WS contract: 12-byte header + one complete H.264 access unit.

    Carry a unique packet PTS through decoding. Do not label buffered pictures
    with the header of the most recently received message. Old/fractured protocols
    fail closed; we do not use the preview's approximate parser metadata.
    """

    def __init__(self):
        self.codec = av.CodecContext.create("h264", "r")
        self.metadata = OrderedDict()
        self.token = 0

    def decode(self, message):
        if not isinstance(message, bytes) or len(message) <= 12:
            raise ValueError("Expected 12-byte header and binary H.264 access unit")
        payload = message[12:]
        if not (payload.startswith(b"\x00\x00\x01") or payload.startswith(b"\x00\x00\x00\x01")):
            raise ValueError(
                "Unsupported camera framing; exact frame alignment requires 12-byte headers"
            )
        timestamp, index = struct.unpack_from("<Qi", message)
        if index < 0:
            return []  # Engine frames produced before indexed rendering started.
        token = self.token
        self.token += 1
        self.metadata[token] = {
            "simulate_index": index,
            "source_timestamp_raw": timestamp,
            "received_monotonic": time.monotonic(),
            "received_wall_ns": time.time_ns(),
            "alignment": "packet_pts_to_decoded_frame",
        }
        while len(self.metadata) > 128:
            self.metadata.popitem(last=False)
        packet = av.Packet(payload)
        packet.pts = token
        packet.time_base = Fraction(1, 1_000_000)
        result = []
        for frame in self.codec.decode(packet):
            meta = self.metadata.pop(frame.pts, None)
            if meta is None:
                raise ValueError(
                    "Decoded picture has no matching packet PTS; refusing approximate alignment"
                )
            result.append((frame.to_ndarray(format="bgr24"), meta))
        return result
