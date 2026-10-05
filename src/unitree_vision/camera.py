"""OrcaLab 26.8.2 camera diagnostics. Q/Esc: quit; S: save original images.

Default mode loads the SDK model (initializing its pose), pauses server physics,
and renders without stepping physics or setting actuator controls. Use --receive-only alongside a controller
that already calls env.render(). This is a preview, not a training data recorder.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import struct
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import av
import cv2
import grpc
import numpy as np
import websockets
from orca_gym.protos import mjc_message_pb2 as pb
from orca_gym.protos.mjc_message_pb2_grpc import GrpcServiceStub

ROLES = {
    "head": ("head_cam", 7090),
    "wrist_r": ("camera_right", 7080),
    "wrist_l": ("camera_left", 7070),
}


def unpack_message(data: bytes) -> tuple[bytes, dict]:
    """Accept Annex-B H.264 with the SDK's 12-byte or legacy 8-byte prefix."""

    def annexb(offset):
        return (
            data[offset : offset + 3] == b"\x00\x00\x01"
            or data[offset : offset + 4] == b"\x00\x00\x00\x01"
        )

    # Prefer new framing, as in the competition camera helper.
    offset = next((n for n in (12, 8, 0) if annexb(n)), None)
    if offset is None:
        raise ValueError("No H.264 Annex-B start code at offset 12, 8 or 0")
    meta = {"header_bytes": offset}
    if offset >= 8:
        meta["source_timestamp_raw"] = struct.unpack_from("<Q", data)[0]
    if offset == 12:
        meta["simulate_index"] = struct.unpack_from("<i", data, 8)[0]
    return data[offset:], meta


def resolve_camera(names, role, explicit=None):
    if explicit:
        if explicit not in names:
            raise ValueError(f"Camera {explicit!r} not registered: {names}")
        return explicit
    token = ROLES[role][0]
    matches = [n for n in names if n == token or n.startswith(token + "_")]
    if token in matches:
        return token
    if len(matches) != 1:
        raise ValueError(
            f"Cannot uniquely resolve {role}: {matches}. Use --{role.replace('_', '-')}-name."
        )
    return matches[0]


class CameraStream:
    """Bounded latest-frame storage; reconnection resets the H.264 decoder."""

    def __init__(self, role, host, port):
        self.role, self.uri = role, f"ws://{host}:{port}"
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.frame = None
        self.metadata = {}
        self.index = 0
        self.error = "waiting for connection"
        self.frame_times = deque(maxlen=120)
        self.thread = threading.Thread(target=self._run, daemon=True, name=role)

    def _run(self):
        asyncio.run(self._receive())

    async def _receive(self):
        while not self.stop_event.is_set():
            try:
                decoder = av.CodecContext.create("h264", "r")
                async with websockets.connect(
                    self.uri,
                    open_timeout=3,
                    close_timeout=1,
                    max_size=16 * 1024 * 1024,
                    max_queue=2,
                ) as ws:
                    with self.lock:
                        self.error = "connected; waiting for keyframe/render"
                    while not self.stop_event.is_set():
                        try:
                            data = await asyncio.wait_for(ws.recv(), timeout=0.5)
                        except TimeoutError:
                            continue
                        if not isinstance(data, bytes):
                            raise ValueError("Expected a binary H.264 WebSocket message")
                        payload, meta = unpack_message(data)
                        with self.lock:
                            self.error = f"received H.264 bytes={len(payload)}; waiting for decoder"
                        # parse() incrementally handles split NALs; no growing BytesIO.
                        for packet in decoder.parse(payload):
                            for decoded in decoder.decode(packet):
                                now = time.monotonic()
                                with self.lock:
                                    self.frame = decoded.to_ndarray(format="bgr24")
                                    self.index += 1
                                    self.frame_times.append(now)
                                    self.metadata = {
                                        **meta,
                                        "received_monotonic": now,
                                        "received_wall_ns": time.time_ns(),
                                        "source_metadata_note": "last contributing WebSocket message; decoder may buffer frames",
                                    }
                                    self.error = ""
            except Exception as exc:
                with self.lock:
                    self.error = f"{type(exc).__name__}: {exc}"
                await asyncio.sleep(0.5)

    def snapshot(self):
        with self.lock:
            fps = (
                (len(self.frame_times) - 1) / (self.frame_times[-1] - self.frame_times[0])
                if len(self.frame_times) > 1 and self.frame_times[-1] > self.frame_times[0]
                else 0
            )
            return (
                None if self.frame is None else self.frame.copy(),
                self.index,
                dict(self.metadata),
                fps,
                self.error,
            )

    def stop(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=5)


def check_response(response, operation):
    if response.status != 0:
        raise RuntimeError(f"{operation}: {response.error_message}")
    return response


class PreviewSession:
    def __init__(self, args, stream_factory=CameraStream):
        self.args = args
        self.stream_factory = stream_factory
        self.channel = grpc.insecure_channel(args.grpc_address)
        self.stub = GrpcServiceStub(self.channel)
        self.original = {}
        self.started = []
        self.streams = {}
        self.env = None
        self.env_loop = None
        self.previous_loop = None
        self._closed = False

    def call(self, method, request):
        return getattr(self.stub, method)(request, timeout=5)

    def open(self):
        grpc.channel_ready_future(self.channel).result(timeout=self.args.timeout)
        if not self.args.receive_only:
            from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv

            try:
                self.previous_loop = asyncio.get_event_loop()
            except RuntimeError:
                self.previous_loop = None
            self.env_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.env_loop)
            print(
                "Loading SDK model; preview initializes the scene pose and pauses server physics.",
                flush=True,
            )
            try:
                self.env = OrcaGymLocalEnv(
                    frame_skip=1,
                    orcagym_addr=self.args.grpc_address,
                    agent_names=[getattr(self.args, "agent", "g1_pick")],
                    time_step=0.001,
                )
                self.env.set_render_fps(round(self.args.fps))
            except Exception:
                self.call("SetSimulationState", pb.SetSimulationStateRequest(state=pb.RUNNING))
                raise
        response = check_response(
            self.call("GetCameraNames", pb.GetCameraNamesRequest()), "GetCameraNames"
        )
        names = list(response.camera_names)
        print("Registered cameras:", names, flush=True)
        for role in self.args.cameras.split(","):
            name = resolve_camera(names, role, getattr(self.args, role + "_name"))
            props = check_response(
                self.call("GetCameraProperties", pb.GetCameraPropertiesRequest(camera_name=name)),
                name,
            )
            port = props.color_port
            if not props.capture_rgb or not 0 < port < 65536:
                raise RuntimeError(
                    f"{name}: enable Color Camera and configure Color Port in OrcaLab first"
                )
            self.original[name] = props.streaming_enabled
            if not props.streaming_enabled:
                check_response(
                    self.call(
                        "SetStreamingEnabled",
                        pb.SetStreamingEnabledRequest(camera_name=name, enabled=True),
                    ),
                    name,
                )
                self.started.append(name)
            print(
                f"{role}: {name}, RGB {props.width}x{props.height}, NVENC={props.use_nvenc}, port={port}",
                flush=True,
            )
            stream = self.stream_factory(role, self.args.host, port)
            self.streams[role] = stream
            stream.thread.start()
        if self.env is not None:
            print("Fixed-pose preview: do not run a robot controller concurrently.", flush=True)

    def render(self, index):
        if self.env is not None:
            self.env.render(
                simulate_index=index, request_idr=(index % max(1, round(self.args.fps)) == 0)
            )

    def close(self):
        if self._closed:
            return
        self._closed = True
        for stream in self.streams.values():
            stream.stop()
        # Leave streams that were already enabled unchanged.
        for name in self.started:
            try:
                check_response(
                    self.call(
                        "SetStreamingEnabled",
                        pb.SetStreamingEnabledRequest(camera_name=name, enabled=False),
                    ),
                    name,
                )
            except Exception as exc:
                print(f"Cleanup warning for {name}: {exc}", flush=True)
        if self.env is not None:
            try:
                self.env.close()
            except Exception as exc:
                print(f"Environment cleanup warning: {exc}", flush=True)
            finally:
                try:
                    self.call("SetSimulationState", pb.SetSimulationStateRequest(state=pb.RUNNING))
                except Exception as exc:
                    print(f"Simulation resume warning: {exc}", flush=True)
        if self.env_loop is not None:
            self.env_loop.close()
            asyncio.set_event_loop(self.previous_loop)
        self.channel.close()


def save_images(output, snapshots):
    folder = output / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    ready = {role: snap for role, snap in snapshots.items() if snap[0] is not None}
    if not ready:
        print("No decoded frames to save", flush=True)
        return
    folder.mkdir(parents=True, exist_ok=False)
    metadata = {}
    for role, (frame, index, meta, fps, error) in ready.items():
        ok, encoded = cv2.imencode(".png", frame)
        if not ok:
            raise RuntimeError(f"PNG encoding failed: {role}")
        # tofile supports Chinese Windows paths, unlike some cv2.imwrite builds.
        encoded.tofile(str(folder / f"{role}.png"))
        metadata[role] = {
            **meta,
            "frame_index": index,
            "shape_hwc": list(frame.shape),
            "fps": fps,
            "error": error,
            "color_order_on_disk": "RGB PNG",
        }
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Saved:", folder.resolve(), flush=True)


def tile(role, snapshot):
    frame, index, meta, fps, error = snapshot
    canvas = np.zeros((410, 640, 3), dtype=np.uint8)
    if frame is not None:
        h, w = frame.shape[:2]
        scale = min(640 / w, 360 / h)
        image = cv2.resize(frame, (max(1, round(w * scale)), max(1, round(h * scale))))
        y, x = (360 - image.shape[0]) // 2, (640 - image.shape[1]) // 2
        canvas[y : y + image.shape[0], x : x + image.shape[1]] = image
        age = time.monotonic() - meta["received_monotonic"]
        label = f"{role}: {w}x{h} FPS={fps:.1f} frame={index} age={age:.2f}s"
        status = error or ("STALE FRAME" if age > 2 else "OK")
    else:
        label, status = f"{role}: no decoded frame", error
    cv2.putText(canvas, label, (8, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
    cv2.putText(canvas, status[:90], (8, 402), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 180, 255), 1)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grpc-address", default="localhost:50051")
    parser.add_argument("--host", default="localhost", help="Camera WebSocket host")
    parser.add_argument(
        "--cameras", default="head,wrist_r", help="Comma-separated head,wrist_r,wrist_l"
    )
    for role in ROLES:
        parser.add_argument("--" + role.replace("_", "-") + "-name", dest=role + "_name")
    parser.add_argument(
        "--receive-only", action="store_true", help="Another controller must drive rendering"
    )
    parser.add_argument("--fps", type=float, default=20)
    parser.add_argument(
        "--timeout", type=float, default=20, help="Connection/first-frame timeout seconds"
    )
    parser.add_argument(
        "--duration", type=float, default=0, help="Exit after N seconds; 0 means until Q/Ctrl+C"
    )
    parser.add_argument(
        "--headless", action="store_true", help="Console validation, no preview window"
    )
    parser.add_argument("--save-on-exit", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("data/camera_test"))
    args = parser.parse_args()
    roles = args.cameras.split(",")
    if any(r not in ROLES for r in roles) or len(set(roles)) != len(roles):
        parser.error("--cameras must contain unique roles from head,wrist_r,wrist_l")
    if (
        not all(math.isfinite(x) for x in (args.fps, args.timeout, args.duration))
        or not 1 <= args.fps <= 60
        or args.timeout <= 0
        or args.duration < 0
    ):
        parser.error("fps must be in [1,60], timeout > 0, duration >= 0")
    session = PreviewSession(args)
    window = "OrcaLab cameras | S save | Q/Esc quit"
    try:
        session.open()
        start = last_report = time.monotonic()
        index = 0
        if not args.headless:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        while True:
            tick = time.monotonic()
            session.render(index)
            index += 1
            snapshots = {role: stream.snapshot() for role, stream in session.streams.items()}
            missing = [role for role, snap in snapshots.items() if snap[0] is None]
            if missing and tick - start > args.timeout:
                detail = {role: snapshots[role][4] for role in missing}
                raise RuntimeError(
                    f"First-frame timeout: {detail}. Check rendering, Color Camera, NVENC and ports."
                )
            stale = [
                role
                for role, snap in snapshots.items()
                if snap[0] is not None and tick - snap[2]["received_monotonic"] > 5
            ]
            if stale:
                raise RuntimeError(f"Stream stopped updating for >5s: {stale}")
            if tick - last_report >= 2:
                print(
                    " | ".join(
                        f"{r}: frames={s[1]} fps={s[3]:.1f} {s[4]}" for r, s in snapshots.items()
                    ),
                    flush=True,
                )
                last_report = tick
            if not args.headless:
                cv2.imshow(
                    window, np.hstack([tile(role, snap) for role, snap in snapshots.items()])
                )
                key = cv2.waitKey(1) & 0xFF
                if (
                    key in (ord("q"), ord("Q"), 27)
                    or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1
                ):
                    break
                if key in (ord("s"), ord("S")):
                    save_images(args.output, snapshots)
            if args.duration and tick - start >= args.duration:
                if missing:
                    raise RuntimeError(
                        f"Test ended without frames from {missing}; increase --duration"
                    )
                if any(snap[1] < 2 for snap in snapshots.values()):
                    raise RuntimeError("Test ended without multiple fresh frames on each camera")
                break
            time.sleep(max(0, 1 / args.fps - (time.monotonic() - tick)))
        if args.save_on_exit:
            save_images(args.output, {r: s.snapshot() for r, s in session.streams.items()})
        print("Camera test completed", flush=True)
        return 0
    except KeyboardInterrupt:
        print("Stopped by user", flush=True)
        return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", flush=True)
        return 1
    finally:
        session.close()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
