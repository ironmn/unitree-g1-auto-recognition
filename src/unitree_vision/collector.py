"""Aligned observation collection; CLI fixed pose or embedded controller mode."""

from __future__ import annotations

import argparse
import logging
import time
from functools import partial
from pathlib import Path

import cv2
import numpy as np

from .camera import ROLES, PreviewSession, check_response, resolve_camera, tile
from .config import CollectorConfig, parse_config
from .dataset import DatasetWriter, atomic_json, runtime_provenance
from .pairing import PairBuffer
from .state import StateReader
from .streams import IndexedCameraStream

logger = logging.getLogger(__name__)


class ObservationCollector:
    """Attach to an existing controller; all submit/poll/save calls on its thread.

    submit(index) captures state BEFORE caller renders the identical index.
    poll() joins decoded frames; it must be called periodically even when idle.
    This class does not create/reset/step the supplied environment.
    """

    def __init__(self, env, args, session=None):
        args = CollectorConfig.from_args(args)
        self.args, self.env = args, env
        self._closed = False
        self.status = "initialized"

        self.session = session
        self.owns_session = session is None
        self.reader = StateReader.from_env(env, agent=args.agent, profile=args.profile)
        self.pairs = PairBuffer(args.cameras.split(","), args.pair_timeout, args.pair_capacity)
        self.recording = args.record
        self.save_next = False
        self.matched = 0
        self.writer = None
        self.last_complete = time.monotonic()
        self.sequence = 0

    def open(self):
        if self._closed or self.writer is not None:
            raise RuntimeError(
                "Collector is closed or already open; create a new collector per run"
            )
        try:
            self._open()
        except BaseException:
            self.close()
            raise

    def _open(self):
        if self.owns_session:
            from dataclasses import replace

            receive_args = replace(self.args, receive_only=True)
            self.session = PreviewSession(
                receive_args,
                stream_factory=partial(IndexedCameraStream, capacity=self.args.camera_buffer),
            )
            self.session.open()
        if set(self.session.streams) != set(self.args.cameras.split(",")):
            raise ValueError("Session streams must match configured camera roles")
        cameras = {}
        from orca_gym.protos import mjc_message_pb2 as pb

        names = list(
            check_response(
                self.session.call("GetCameraNames", pb.GetCameraNamesRequest()), "GetCameraNames"
            ).camera_names
        )
        for role in self.session.streams:
            name = resolve_camera(names, role, getattr(self.args, role + "_name"))
            p = check_response(
                self.session.call(
                    "GetCameraProperties", pb.GetCameraPropertiesRequest(camera_name=name)
                ),
                name,
            )
            cameras[role] = {
                "name": name,
                "width": p.width,
                "height": p.height,
                "port": p.color_port,
            }
        self.writer = DatasetWriter(
            self.args.output,
            {
                "schema_version": 1,
                "resolved_config": self.args.snapshot(),
                "provenance": runtime_provenance(),
                "state_manifest": self.reader.manifest,
                "cameras": cameras,
                "prompt": self.args.prompt,
                "alignment": "exact simulate_index through packet PTS",
                "mode": "controller_integration" if self.owns_session else "fixed_pose",
                "physics_stepped_by_collector": False,
                "has_actions": False,
                "pair_timeout_s": self.args.pair_timeout,
                "sample_every": self.args.sample_every,
            },
        )
        self.status = "open"
        logger.info("Dataset: %s", self.writer.folder.resolve())

    def submit(self, index, prompt=None):
        self._require_open()
        state = self.reader.read(sequence=self.sequence, simulate_index=index)
        self.pairs.submit(state, self.args.prompt if prompt is None else prompt)
        self.sequence += 1

    def poll(self):
        self._require_open()
        for role, stream in self.session.streams.items():
            for frame, meta in stream.drain():
                self.pairs.feed(role, frame, meta)
        ready = self.pairs.poll()
        for entry in ready:
            self.matched += 1
            self.last_complete = time.monotonic()
            if self.save_next or (
                self.recording and (self.matched - 1) % self.args.sample_every == 0
            ):
                self.writer.save(entry)
                self.save_next = False
        return ready

    def _require_open(self):
        if self.writer is None or self._closed:
            raise RuntimeError("Collector must be open")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, kind, value, traceback):
        if kind is not None:
            self.status = "failed"
        self.close()

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self.writer:
                atomic_json(
                    self.writer.folder / "summary.json",
                    {
                        "status": "closed" if self.status == "open" else self.status,
                        "saved": self.writer.count,
                        "matched": self.matched,
                        "dropped_incomplete": self.pairs.dropped,
                        "unfinished": len(self.pairs.pending),
                        "camera_queue_overflow": {
                            r: s.overflow for r, s in self.session.streams.items()
                        },
                    },
                )
        except Exception:
            logger.exception("Failed to write summary; committed samples remain recoverable")
        finally:
            if self.owns_session and self.session:
                self.session.close()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, help="TOML [collector] settings; CLI overrides")
    p.add_argument("--agent", default="g1_pick")
    p.add_argument("--pair-capacity", type=int, default=64)
    p.add_argument("--camera-buffer", type=int, default=16)
    p.add_argument("--grpc-address", default="localhost:50051")
    p.add_argument("--host", default="localhost")
    p.add_argument("--cameras", default="head,wrist_r")
    for role in ROLES:
        p.add_argument("--" + role.replace("_", "-") + "-name", dest=role + "_name")
    p.add_argument("--profile", choices=["full", "arms", "right"], default="full")
    p.add_argument("--prompt", default="识别面板上的按钮4")
    p.add_argument("--fps", type=float, default=10)
    p.add_argument("--timeout", type=float, default=20)
    p.add_argument("--pair-timeout", type=float, default=3)
    p.add_argument("--duration", type=float, default=0)
    p.add_argument(
        "--record",
        action=argparse.BooleanOptionalAction,
        help="Start continuous recording immediately",
    )
    p.add_argument("--sample-every", type=int, default=1, help="Save every N complete pairs")
    p.add_argument("--headless", action=argparse.BooleanOptionalAction)
    p.add_argument("--output", type=Path, default=Path("data/observations"))
    p.set_defaults(receive_only=False, record=False, headless=False)
    return p


def main(argv=None):
    p = parser()
    args = parse_config(p, argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    session = PreviewSession(
        args, stream_factory=partial(IndexedCameraStream, capacity=args.camera_buffer)
    )
    collector = None
    window = "Observations | S save next | R record | Q quit"
    try:
        session.open()
        collector = ObservationCollector(session.env, args, session)
        collector.open()
        # Allow both WS subscriptions to connect before the first indexed render.
        deadline = time.monotonic() + args.timeout
        while not all(s.connected for s in session.streams.values()):
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"Camera connection timeout: {[s.error for s in session.streams.values()]}"
                )
            time.sleep(0.05)
        start = report = time.monotonic()
        collector.last_complete = start
        if not args.headless:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        # Avoid matching residual frames from an earlier index=0 session.
        index = int(time.time() * 1000) % 1_000_000_000
        while True:
            tick = time.monotonic()
            collector.submit(index)
            # Force exactly one render with this index. Each IDR is self-contained;
            # lower FPS is intentional to keep loss/reconnection recoverable.
            session.env.render(simulate_index=index, request_idr=True)
            index += 1
            collector.poll()
            if time.monotonic() - collector.last_complete > args.timeout:
                raise RuntimeError(
                    f"No complete aligned observation for {args.timeout}s: "
                    f"{[s.error for s in session.streams.values()]}"
                )
            if tick - report >= 2:
                print(
                    f"matched={collector.matched} saved={collector.writer.count} "
                    f"dropped={collector.pairs.dropped} recording={collector.recording}",
                    flush=True,
                )
                report = tick
            if not args.headless:
                preview = np.hstack([tile(r, s.snapshot()) for r, s in session.streams.items()])
                cv2.putText(
                    preview,
                    f"REC={collector.recording} saved={collector.writer.count} "
                    f"aligned={collector.matched} waiting_S={collector.save_next}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )
                cv2.imshow(window, preview)
                key = cv2.waitKey(1) & 0xFF
                if (
                    key in (ord("q"), ord("Q"), 27)
                    or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1
                ):
                    break
                if key in (ord("s"), ord("S")):
                    collector.save_next = True
                if key in (ord("r"), ord("R")):
                    collector.recording = not collector.recording
            if args.duration and tick - start >= args.duration:
                break
            time.sleep(max(0, 1 / args.fps - (time.monotonic() - tick)))
        # Drain delayed network/decoder results without changing the last pose.
        drain_until = time.monotonic() + args.pair_timeout
        while collector.pairs.pending and time.monotonic() < drain_until:
            collector.poll()
            time.sleep(0.02)
        collector.poll()
        if not collector.matched or (args.record and not collector.writer.count):
            raise RuntimeError("No aligned samples captured; check camera headers/rendering")
        collector.status = "completed"
        print(f"Completed: saved={collector.writer.count}, matched={collector.matched}", flush=True)
        return 0
    except KeyboardInterrupt:
        if collector:
            collector.status = "interrupted"
        print("Stopped; committed sample folders preserved.", flush=True)
        return 0
    except Exception as exc:
        if collector:
            collector.status = "failed"
        print(f"ERROR: {type(exc).__name__}: {exc}", flush=True)
        return 1
    finally:
        try:
            if collector:
                collector.close()
        finally:
            session.close()
            if not args.headless:
                cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
