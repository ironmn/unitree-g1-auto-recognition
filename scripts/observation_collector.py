"""Paired camera/state recorder for OrcaLab 26.8.2.

CLI owns a fixed-pose SDK environment (resets scene, no physics stepping).
Use ObservationCollector(env, args) inside a controller for moving trajectories.
S: save next complete observation; R: toggle recording; Q/Esc: quit.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import OrderedDict, deque
from datetime import datetime
from fractions import Fraction
import json
from pathlib import Path
import struct
import time

import av
import cv2
import numpy as np
import websockets

try:
    from .camera_preview import CameraStream, PreviewSession, ROLES, tile, resolve_camera, check_response
    from .robot_state import StateReader
except ImportError:
    from camera_preview import CameraStream, PreviewSession, ROLES, tile, resolve_camera, check_response
    from robot_state import StateReader


class IndexedDecoder:
    """Engine WS contract: 12-byte header + one complete H.264 access unit.

    Carry a unique packet PTS through decoding. Do not label buffered pictures
    with the header of the most recently received message. Old/fractured protocols
    fail closed; we do not use the preview's approximate parser metadata.
    """
    def __init__(self):
        self.codec = av.CodecContext.create('h264', 'r')
        self.metadata = OrderedDict()
        self.token = 0

    def decode(self, message):
        if not isinstance(message, bytes) or len(message) <= 12:
            raise ValueError('Expected 12-byte header and binary H.264 access unit')
        payload = message[12:]
        if not (payload.startswith(b'\x00\x00\x01') or payload.startswith(b'\x00\x00\x00\x01')):
            raise ValueError('Unsupported camera framing; exact frame alignment requires 12-byte headers')
        timestamp, index = struct.unpack_from('<Qi', message)
        if index < 0:
            return []  # Engine frames produced before indexed rendering started.
        token = self.token
        self.token += 1
        self.metadata[token] = {'simulate_index': index, 'source_timestamp_raw': timestamp,
                                'received_monotonic': time.monotonic(),
                                'received_wall_ns': time.time_ns(),
                                'alignment': 'packet_pts_to_decoded_frame'}
        while len(self.metadata) > 128:
            self.metadata.popitem(last=False)
        packet = av.Packet(payload)
        packet.pts = token
        packet.time_base = Fraction(1, 1_000_000)
        result = []
        for frame in self.codec.decode(packet):
            meta = self.metadata.pop(frame.pts, None)
            if meta is None:
                raise ValueError('Decoded picture has no matching packet PTS; refusing approximate alignment')
            result.append((frame.to_ndarray(format='bgr24'), meta))
        return result


class IndexedCameraStream(CameraStream):
    def __init__(self, role, host, port):
        super().__init__(role, host, port)
        self.frames = deque(maxlen=16)
        self.connected = False
        self.overflow = 0

    async def _receive(self):
        while not self.stop_event.is_set():
            try:
                decoder = IndexedDecoder()
                async with websockets.connect(self.uri, open_timeout=3, close_timeout=1,
                                               max_size=16 * 1024 * 1024, max_queue=2) as ws:
                    with self.lock:
                        self.connected = True
                        self.error = 'connected; waiting for indexed keyframe'
                    while not self.stop_event.is_set():
                        try:
                            message = await asyncio.wait_for(ws.recv(), timeout=.5)
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
                                self.error = ''
            except Exception as exc:
                with self.lock:
                    self.error = f'{type(exc).__name__}: {exc}'
                await asyncio.sleep(.5)
            finally:
                with self.lock:
                    self.connected = False

    def drain(self):
        with self.lock:
            frames = list(self.frames)
            self.frames.clear()
            return frames


class PairBuffer:
    """Bounded exact-index join; missing camera frames are never substituted."""
    def __init__(self, roles, timeout=3, capacity=64):
        self.roles = tuple(roles)
        self.timeout, self.capacity = timeout, capacity
        self.pending = OrderedDict()
        self.dropped = 0
        self.last_index = -1

    def submit(self, state, prompt):
        index = state['simulate_index']
        if not isinstance(index, int) or index <= self.last_index:
            raise ValueError('simulate_index must be a strictly increasing nonnegative integer')
        self.last_index = index
        self.pending[index] = {'state': state, 'prompt': prompt, 'images': {}, 'created': time.monotonic()}
        if len(self.pending) > self.capacity:
            self.pending.popitem(last=False)
            self.dropped += 1

    def feed(self, role, frame, meta):
        if role not in self.roles:
            raise ValueError(f'Unknown camera role: {role}')
        entry = self.pending.get(meta['simulate_index'])
        if entry is not None and role not in entry['images']:
            entry['images'][role] = (frame, dict(meta))

    def poll(self, now=None):
        now = time.monotonic() if now is None else now
        ready = []
        # Process in order: a delayed earlier camera must not reorder the dataset.
        while self.pending:
            index, entry = next(iter(self.pending.items()))
            if all(r in entry['images'] for r in self.roles):
                ready.append(entry)
                del self.pending[index]
            elif now - entry['created'] >= self.timeout:
                del self.pending[index]
                self.dropped += 1
            else:
                break
        return ready


class DatasetWriter:
    """A sample becomes visible only after all PNGs and its JSON are written."""
    def __init__(self, output, manifest):
        self.folder = Path(output) / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        (self.folder / 'samples').mkdir(parents=True, exist_ok=False)
        self.count = 0
        (self.folder / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False,
                                                           indent=2, allow_nan=False), encoding='utf-8')

    def save(self, entry):
        index = entry['state']['simulate_index']
        sample_id = f'{self.count:06d}_{index:010d}'
        temporary = self.folder / 'samples' / ('.partial_' + sample_id)
        final = self.folder / 'samples' / sample_id
        temporary.mkdir(exist_ok=False)
        images = {}
        for role, (frame, meta) in entry['images'].items():
            if meta['simulate_index'] != index:
                raise ValueError('Image/state frame mismatch')
            ok, encoded = cv2.imencode('.png', frame)
            if not ok:
                raise RuntimeError(f'PNG encoding failed: {role}')
            encoded.tofile(str(temporary / f'{role}.png'))
            images[role] = dict(meta, file=f'{role}.png', shape_hwc=list(frame.shape),
                                color_order_on_disk='RGB PNG')
        record = {'sample_id': sample_id, 'prompt': entry['prompt'],
                  'observation': entry['state'], 'images': images,
                  'action': None, 'annotation': None}
        (temporary / 'observation.json').write_text(json.dumps(record, ensure_ascii=False,
                                                             indent=2, allow_nan=False), encoding='utf-8')
        temporary.rename(final)
        # sample folders are the source of truth; index can be rebuilt after a crash.
        with (self.folder / 'index.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps({'sample_id': sample_id, 'simulate_index': index,
                                'file': f'samples/{sample_id}/observation.json'}) + '\n')
        self.count += 1
        return final


class ObservationCollector:
    """Attach to an existing controller; all submit/poll/save calls on its thread.

    submit(index) captures state BEFORE caller renders the identical index.
    poll() joins decoded frames; it must be called periodically even when idle.
    This class does not create/reset/step the supplied environment.
    """
    def __init__(self, env, args, session=None):
        self.args, self.env = args, env
        self.session = session
        self.owns_session = session is None
        self.reader = StateReader.from_env(env, agent='g1_pick', profile=args.profile)
        self.pairs = PairBuffer(args.cameras.split(','), args.pair_timeout)
        self.recording = args.record
        self.save_next = False
        self.matched = 0
        self.writer = None
        self.last_complete = time.monotonic()
        self.sequence = 0

    def open(self):
        if self.owns_session:
            import copy
            receive_args = copy.copy(self.args)
            receive_args.receive_only = True
            self.session = PreviewSession(receive_args, stream_factory=IndexedCameraStream)
            self.session.open()
        cameras = {}
        from orca_gym.protos import mjc_message_pb2 as pb
        names = list(check_response(self.session.call('GetCameraNames', pb.GetCameraNamesRequest()),
                                    'GetCameraNames').camera_names)
        for role in self.session.streams:
            name = resolve_camera(names, role, getattr(self.args, role + '_name'))
            p = check_response(self.session.call('GetCameraProperties',
                               pb.GetCameraPropertiesRequest(camera_name=name)), name)
            cameras[role] = {'name': name, 'width': p.width, 'height': p.height, 'port': p.color_port}
        self.writer = DatasetWriter(self.args.output, {
            'schema_version': 1, 'state_manifest': self.reader.manifest, 'cameras': cameras,
            'prompt': self.args.prompt, 'alignment': 'exact simulate_index through packet PTS',
            'mode': 'controller_integration' if self.owns_session else 'fixed_pose',
            'physics_stepped_by_collector': False, 'has_actions': False,
            'pair_timeout_s': self.args.pair_timeout, 'sample_every': self.args.sample_every})
        print(f'Dataset: {self.writer.folder.resolve()}', flush=True)

    def submit(self, index, prompt=None):
        state = self.reader.read(sequence=self.sequence, simulate_index=index)
        self.pairs.submit(state, self.args.prompt if prompt is None else prompt)
        self.sequence += 1

    def poll(self):
        for role, stream in self.session.streams.items():
            for frame, meta in stream.drain():
                self.pairs.feed(role, frame, meta)
        ready = self.pairs.poll()
        for entry in ready:
            self.matched += 1
            self.last_complete = time.monotonic()
            if self.save_next or (self.recording and (self.matched - 1) % self.args.sample_every == 0):
                self.writer.save(entry)
                self.save_next = False
        return ready

    def close(self):
        if self.writer:
            (self.writer.folder / 'summary.json').write_text(json.dumps({
                'saved': self.writer.count, 'matched': self.matched,
                'dropped_incomplete': self.pairs.dropped, 'unfinished': len(self.pairs.pending),
                'camera_queue_overflow': {r: s.overflow for r, s in self.session.streams.items()}
            }, indent=2), encoding='utf-8')
        if self.owns_session and self.session:
            self.session.close()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--grpc-address', default='localhost:50051')
    p.add_argument('--host', default='localhost')
    p.add_argument('--cameras', default='head,wrist_r')
    for role in ROLES:
        p.add_argument('--' + role.replace('_', '-') + '-name', dest=role + '_name')
    p.add_argument('--profile', choices=['full', 'arms', 'right'], default='full')
    p.add_argument('--prompt', default='识别面板上的按钮4')
    p.add_argument('--fps', type=float, default=10)
    p.add_argument('--timeout', type=float, default=20)
    p.add_argument('--pair-timeout', type=float, default=3)
    p.add_argument('--duration', type=float, default=0)
    p.add_argument('--record', action='store_true', help='Start continuous recording immediately')
    p.add_argument('--sample-every', type=int, default=1, help='Save every N complete pairs')
    p.add_argument('--headless', action='store_true')
    p.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'data' / 'observations')
    p.set_defaults(receive_only=False)
    return p


def main():
    p = parser()
    args = p.parse_args()
    roles = args.cameras.split(',')
    if not roles or any(r not in ROLES for r in roles) or len(set(roles)) != len(roles):
        p.error('cameras must be unique head,wrist_r,wrist_l roles')
    if not all(np.isfinite(x) for x in [args.fps, args.timeout, args.pair_timeout, args.duration]) or not (
            1 <= args.fps <= 30 and args.timeout > 0 and args.pair_timeout > 0
            and args.duration >= 0 and args.sample_every > 0):
        p.error('fps in [1,30], timeouts > 0, duration >= 0, sample-every >= 1; values finite')
    if args.headless and not args.record:
        p.error('headless mode requires --record')
    session = PreviewSession(args, stream_factory=IndexedCameraStream)
    collector = None
    window = 'Observations | S save next | R record | Q quit'
    try:
        session.open()
        collector = ObservationCollector(session.env, args, session)
        collector.open()
        # Allow both WS subscriptions to connect before the first indexed render.
        deadline = time.monotonic() + args.timeout
        while not all(s.connected for s in session.streams.values()):
            if time.monotonic() > deadline:
                raise RuntimeError(f'Camera connection timeout: {[s.error for s in session.streams.values()]}')
            time.sleep(.05)
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
            session.env.loop.run_until_complete(session.env.gym.render(index, True))
            index += 1
            collector.poll()
            if time.monotonic() - collector.last_complete > args.timeout:
                raise RuntimeError(f'No complete aligned observation for {args.timeout}s: '
                                   f'{[s.error for s in session.streams.values()]}')
            if tick - report >= 2:
                print(f'matched={collector.matched} saved={collector.writer.count} '
                      f'dropped={collector.pairs.dropped} recording={collector.recording}', flush=True)
                report = tick
            if not args.headless:
                preview = np.hstack([tile(r, s.snapshot()) for r, s in session.streams.items()])
                cv2.putText(preview, f'REC={collector.recording} saved={collector.writer.count} '
                            f'aligned={collector.matched} waiting_S={collector.save_next}',
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
                cv2.imshow(window, preview)
                key = cv2.waitKey(1) & 0xff
                if key in (ord('q'), ord('Q'), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key in (ord('s'), ord('S')):
                    collector.save_next = True
                if key in (ord('r'), ord('R')):
                    collector.recording = not collector.recording
            if args.duration and tick - start >= args.duration:
                break
            time.sleep(max(0, 1 / args.fps - (time.monotonic() - tick)))
        # Drain delayed network/decoder results without changing the last pose.
        drain_until = time.monotonic() + args.pair_timeout
        while collector.pairs.pending and time.monotonic() < drain_until:
            collector.poll()
            time.sleep(.02)
        collector.poll()
        if not collector.matched or (args.record and not collector.writer.count):
            raise RuntimeError('No aligned samples captured; check camera headers/rendering')
        print(f'Completed: saved={collector.writer.count}, matched={collector.matched}', flush=True)
        return 0
    except KeyboardInterrupt:
        print('Stopped; committed sample folders preserved.', flush=True)
        return 0
    except Exception as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}', flush=True)
        return 1
    finally:
        try:
            if collector:
                collector.close()
        finally:
            session.close()
            if not args.headless:
                cv2.destroyAllWindows()


if __name__ == '__main__':
    raise SystemExit(main())
